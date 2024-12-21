import torch.nn

import Spline
import nerf


def init_linear_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            torch.nn.init.xavier_normal_(m.weight, 0.1)
        else:
            torch.nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)


class Model(nerf.Model):
    def __init__(self, se3_start, se3_end):
        super().__init__()
        self.start = se3_start
        self.end = se3_end

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        self.graph.se3 = torch.nn.Embedding(self.start.shape[0], 6 * 2)

        # self.graph.linear = torch.nn.Linear(90 * args.deblur_images * self.start.shape[0], )   # #######

        start_end = torch.cat([self.start, self.end], -1)  # (29,12)
        self.graph.se3.weight.data = torch.nn.Parameter(start_end)

        num_wide = 64
        num_hidden = 3
        short_cut = False

        in_cnl_rgb = args.N_rgb_rand * 3
        # in_cnl = 90
        out_cnl = 1
        hiddens = [torch.nn.Linear(num_wide, num_wide) if i % 2 == 0 else torch.nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        # hiddens = [nn.Linear(num_wide, num_wide), nn.ReLU()] * num_hidden

        self.linears1 = torch.nn.ModuleList()
        self.linears2 = torch.nn.ModuleList()

        # 假设 args.i_train 在 Model 被实例化的作用域内可用。
        for i in range(len(args.i_train)):
            self.linears1.append(torch.nn.Sequential(
                torch.nn.Linear(in_cnl_rgb, num_wide), torch.nn.ReLU(),
                *hiddens,
            ))
            self.linears2.append(torch.nn.Sequential(
                torch.nn.Linear((num_wide + in_cnl_rgb) if short_cut else num_wide, num_wide), torch.nn.ReLU(),
                torch.nn.Linear(num_wide, out_cnl)
            ))

        in_cnl_depth = args.N_depth_rand
        self.linears3 = torch.nn.ModuleList()
        self.linears4 = torch.nn.ModuleList()

        for i in range(len(args.i_train)):
            self.linears3.append(torch.nn.Sequential(
                torch.nn.Linear(in_cnl_depth, num_wide), torch.nn.ReLU(),
                *hiddens,
            ))
            self.linears4.append(torch.nn.Sequential(
                torch.nn.Linear((num_wide + in_cnl_depth) if short_cut else num_wide, num_wide), torch.nn.ReLU(),
                torch.nn.Linear(num_wide, out_cnl)
            ))

        self.linears1.apply(init_linear_weights)
        self.linears2.apply(init_linear_weights)
        self.linears3.apply(init_linear_weights)
        self.linears4.apply(init_linear_weights)

        return self.graph

    def setup_optimizer(self, args):

        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        grad_vars_rgb = list(self.linears1.parameters()) + list(self.linears2.parameters())
        self.optim_rgb = torch.optim.Adam(params=grad_vars_rgb, lr=args.lrate)

        grad_vars_depth = list(self.linears3.parameters()) + list(self.linears4.parameters())
        self.optim_depth = torch.optim.Adam(params=grad_vars_depth, lr=args.lrate)

        return self.optim, self.optim_se3, self.optim_rgb, self.optim_depth

    def Simulate_bulr_rgb(self, rgb_blur, extras_blur, args):
        rgb_blur0 = rgb_blur.reshape(len(args.i_train), args.deblur_images, -1)
        rgb_blur_list = []
        extras_blur_list = []
        for i in range(len(args.i_train)):
            x1 = rgb_blur0[i]
            x1 = self.linears1[i](x1)
            weight = self.linears2[i](x1)
            weight = torch.softmax(weight, dim=0)
            rgb_blur_ = torch.sum(rgb_blur[i] * weight[..., None], dim=0)
            rgb_blur_list.append(rgb_blur_)
            extras_blur_ = torch.sum(extras_blur[i] * weight[..., None], dim=0)
            extras_blur_list.append(extras_blur_)

        rgb_blur = torch.stack(rgb_blur_list, 0)
        rgb_blur = rgb_blur.reshape(-1, 3)
        extras_blur = torch.stack(extras_blur_list, 0)
        extras_blur = extras_blur.reshape(-1, 3)
        return rgb_blur, extras_blur

    def Simulate_bulr_depth(self, depths_blur, args):
        depths_blur0 = depths_blur
        depths_blur_list = []
        for i in range(len(args.i_train)):
            x1 = depths_blur0[i]
            x1 = self.linears3[i](x1)
            weight = self.linears4[i](x1)
            weight = torch.softmax(weight, dim=0)
            depths_blur_ = torch.sum(depths_blur[i] * weight[...], dim=0)
            depths_blur_list.append(depths_blur_)
        depths_blur = torch.stack(depths_blur_list, 0)
        depths_blur = depths_blur.reshape(-1)
        return depths_blur


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)
        self.se3_start = None
        self.se3_end = None

    def get_pose(self, i, img_idx, args):
        se3_start = self.se3.weight[:, :6][img_idx]
        se3_end = self.se3.weight[:, 6:][img_idx]
        pose_nums = torch.arange(args.deblur_images).reshape(1, -1).repeat(se3_start.shape[0], 1)
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1, args.deblur_images)

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, args.deblur_images)
        return spline_poses

    def get_pose_even(self, i, img_idx, num):
        deblur_images_num = num + 1
        se3_start = self.se3.weight[:, :6][img_idx]
        se3_end = self.se3.weight[:, 6:][img_idx]
        pose_nums = torch.arange(deblur_images_num).reshape(1, -1).repeat(se3_start.shape[0], 1)
        seg_pos_x = torch.arange(se3_start.shape[0]).reshape([se3_start.shape[0], 1]).repeat(1, deblur_images_num)

        se3_start = se3_start[seg_pos_x, :]
        se3_end = se3_end[seg_pos_x, :]

        spline_poses = Spline.SplineN_linear(se3_start, se3_end, pose_nums, deblur_images_num)
        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
