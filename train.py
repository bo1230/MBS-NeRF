import torch

from nerf import *
import optimize_pose_linear, optimize_pose_cubic
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt
from load_llff import load_llff_data, load_colmap_depth
from metrics import compute_img_metric
import novel_view_test

from data import RayDataset, DepthDataset
from torch.utils.data import DataLoader

def train():
    parser = config_parser()
    args = parser.parse_args()
    print('spline numbers: ', args.deblur_images)
    torch.cuda.set_device(args.gpu_id)

    imgs_sharp_dir = os.path.join(args.datadir, 'images_test')
    imgs_sharp = load_imgs(imgs_sharp_dir)

    # Load data images and groundtruth
    K = None
    if args.dataset_type == 'llff':
        if args.colmap_depth:
            depth_gts = load_colmap_depth(args.datadir, factor=args.factor, bd_factor=.75)  # 每张图像的有效点的深度，图像坐标，误差

        images_all, poses_all, bds_start, render_poses = load_llff_data(args.datadir, pose_state=None,
                                                                      factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify)
        hwf = poses_all[0, :3, -1]

        # split train/val/test
        i_test = torch.arange(0, images_all.shape[0], args.llffhold)
        i_val = i_test
        i_train = torch.tensor(args.i_train).long()  # 3v
        # i_train = torch.tensor([1, 17, 25]).long()
        # i_train = torch.tensor([1, 9, 17, 25, 31]).long()
        # i_train = torch.tensor([1, 6, 11, 15, 19, 23, 27, 31]).long()  # 8v

        # train data
        images = images_all[i_train]
        poses_start = poses_all[i_train]
        poses_test = poses_all[i_test]

        # novel view data
        if args.novel_view:
            images_novel = images_all[i_test]
        # gt data
        imgs_sharp_train = imgs_sharp[i_train]
        imgs_sharp_test = imgs_sharp[i_test]

        # get poses
        poses_end = poses_start
        poses_start_se3 = SE3_to_se3_N(poses_start[:, :3, :4])
        poses_end_se3 = poses_start_se3
        poses_org = poses_start.repeat(args.deblur_images, 1, 1)
        poses = poses_org[:, :, :4]
        # print('poses_end: ', poses_end)
        # print('poses_end_se3 = poses_start_se3: ', poses_start_se3)

        #get test pose
        poses_test = SE3_to_se3_N(poses_test[:, :3, :4])

        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = torch.min(bds_start) * .9
            far = torch.max(bds_start) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    test_metric_file_novel = os.path.join(basedir, expname, 'test_metrics_novel.txt')
    test_file = os.path.join(basedir, expname, 'test.txt')
    print_file = os.path.join(basedir, expname, 'print.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    if args.load_weights:
        if args.linear:
            print('Linear Spline Model Loading!')
            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)
        else:
            print('Cubic Spline Model Loading!')
            model = optimize_pose_cubic.Model(poses_start_se3, poses_start_se3, poses_start_se3, poses_start_se3)
        graph = model.build_network(args)
        optimizer, optimizer_se3, optimizer_rgb, optimizer_depth = model.setup_optimizer(args)
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))  # here
        graph_ckpt = torch.load(path)
        graph.load_state_dict(graph_ckpt['graph'])
        optimizer.load_state_dict(graph_ckpt['optimizer'])
        optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
        optimizer_rgb.load_state_dict(graph_ckpt['optimizer_rgb'])
        optimizer_depth.load_state_dict(graph_ckpt['optimizer_depth'])
        global_step = graph_ckpt['global_step']

    else:
        if args.linear:
            low, high = 0.0001, 0.005
            rand = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_start_se3 = poses_start_se3 + rand

            model = optimize_pose_linear.Model(poses_start_se3, poses_end_se3)  # 29个初始位姿和结束位姿
        else:
            low, high = 0.0001, 0.01
            rand1 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand2 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            rand3 = (high - low) * torch.rand(poses_start_se3.shape[0], 6) + low
            poses_se3_1 = poses_start_se3 + rand1
            poses_se3_2 = poses_start_se3 + rand2
            poses_se3_3 = poses_start_se3 + rand3

            model = optimize_pose_cubic.Model(poses_start_se3, poses_se3_1, poses_se3_2, poses_se3_3)

        graph = model.build_network(args)  # nerf, nerf_fine, forward
        optimizer, optimizer_se3, optimizer_rgb, optimizer_depth = model.setup_optimizer(args)


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    if not args.load_weights:
        global_step = start
    global_step_ = global_step
    threshold = N_iters + 1

    poses_num = poses.shape[0]

    for i in trange(start, threshold):
    ### core optimization loop ###
        i = i+global_step_
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)

        img_idx = torch.randperm(images.shape[0])       # 随机化图像索引
        print('img_idx', img_idx)
        #
        if (i % args.i_img == 0 or i % args.i_novel_view == 0) and i > 0:       # i_img = 25,000, i_novel_view = 200,000
            ret, ray_rgb_idx, rays_depths, rays_weights, spline_poses, all_poses = graph.forward(i, depth_gts, img_idx, poses_num, H, W, K,near,far, args)
        else:
            ret, ray_rgb_idx, rays_depths, rays_weights, spline_poses = graph.forward(i, depth_gts, img_idx, poses_num, H, W, K, near, far, args)

        # get image ground truth

        N = len(i_train) * args.deblur_images * len(ray_rgb_idx)

        rgb = ret['rgb_map'][ :N, :]
        disp = ret['disp_map'][ :N]
        acc = ret['acc_map'][:N]
        depth, depth_col = ret['depth_map'][:N], ret['depth_map'][N:]
        if 'rgb0' in ret:
            extras = ret['rgb0'][:N]


        target_s = images[img_idx].reshape(-1, H * W, 3)
        target_s = target_s[:, ray_rgb_idx]
        target_s = target_s.reshape(-1, 3)

        # get depth ground truth
        if args.colmap_depth:
            if args.ndc:
                target_depth = rays_depths / torch.max(bds_start)
            else:
                target_depth = rays_depths

        # average

        shape0 = img_idx.shape[0]
        interval = args.N_rgb_rand
        rgb_list = []
        rgb_list_ = []
        extras_list = []
        extras_list_ = []

        extras_ = 0
        # rgb average
        for j in range(0, shape0 * args.deblur_images):
            rgb_ = rgb[j * interval:(j + 1) * interval]  # (24, 3)
            rgb_list_.append(rgb_)
            if 'rgb0' in ret:
                extras_ = extras[j * interval:(j + 1) * interval]
                extras_list_.append(extras_)
            if (j + 1) % args.deblur_images == 0:
                rgb_list_ = torch.stack(rgb_list_, 0)
                rgb_list.append(rgb_list_)
                rgb_list_ = []
                if 'rgb0' in ret:
                    extras_list_ = torch.stack(extras_list_, 0)
                    extras_list.append(extras_list_)
                    extras_list_ = []

        if 'rgb0' in ret:
            extras_blur = torch.stack(extras_list, 0)


        rgb_blur = torch.stack(rgb_list, 0)   # (deblur_img_num ,img_num, ray_num, 3)
        rgb_blur, extras_blur = model.Simulate_bulr_rgb(rgb_blur, extras_blur, args)

        optimizer_se3.zero_grad()
        optimizer.zero_grad()
        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        img_loss = img2mse(rgb_blur, target_s)

        # depth average
        depth_loss = 0
        depth_ = 0
        depth_list = []
        depth_list_ = []
        interval1 = args.N_depth_rand
        if args.use_depth_loss:
            for j in range(0, shape0 * args.deblur_images):
                depth_ = depth_col[j * interval1:(j + 1) * interval1]  # (24, 3)
                depth_list_.append(depth_)
                if (j + 1) % args.deblur_images == 0:
                    depth_list_ = torch.stack(depth_list_, 0)
                    depth_list.append(depth_list_)
                    depth_list_ = []
            depths_blur = torch.stack(depth_list, 0)
            depths_blur = model.Simulate_bulr_depth(depths_blur, args)

            if args.weighted_loss:
                depth_loss = torch.mean(((depths_blur - target_depth) ** 2) * rays_weights)
            elif args.relative_loss:
                depth_loss = torch.mean(((depths_blur - target_depth) / target_depth)**2)
            else:
                depth_loss = img2mse(depths_blur, target_depth)

        # backward

        loss = img_loss + args.depth_lambda * depth_loss

        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            img_loss0 = img2mse(extras_blur, target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()

        optimizer.step()
        optimizer_se3.step()
        optimizer_rgb.step()
        optimizer_depth.step()


        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000       # 200  250
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))    # lrate 5e-4
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = 0.01
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (global_step / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose

        decay_rate_rgb = 0.01
        new_lrate_rgb = args.blur_rgb_lrate * (decay_rate_rgb ** (global_step / decay_steps))
        for param_group in optimizer_rgb.param_groups:
            param_group['lr'] = new_lrate_rgb

        decay_rate_depth = 0.01
        new_lrate_depth = args.blur_depth_lrate * (decay_rate_depth ** (global_step / decay_steps))
        for param_group in optimizer_depth.param_groups:
            param_group['lr'] = new_lrate_depth
        ###############################

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}")
            with open(print_file, 'a') as outfile:
                outfile.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}\n")

        if i < 10:
            print('coarse_loss:', img_loss0.item())
            with open(print_file, 'a') as outfile:
                outfile.write(f"coarse loss: {img_loss0.item()}\n")

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
                'optimizer_rgb': optimizer_rgb.state_dict(),
                'optimizer_depth': optimizer_depth.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_img == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                if args.deblur_images % 2 == 0:
                    i_render = torch.arange(i_train.shape[0]) * (args.deblur_images+1) + args.deblur_images // 2
                else:
                    i_render = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                imgs_render = render_image_test(i, graph, all_poses[i_render], H, W, K, near, far, args)
            mse_render = compute_img_metric(imgs_sharp_train, imgs_render, 'mse')
            psnr_render = compute_img_metric(imgs_sharp_train, imgs_render, 'psnr')
            ssim_render = compute_img_metric(imgs_sharp_train, imgs_render, 'ssim')
            lpips_render = compute_img_metric(imgs_sharp_train, imgs_render, 'lpips')
            with open(test_metric_file, 'a') as outfile:
                outfile.write(f"iter{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                              f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

        if i % args.i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H, W, K, near, far, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % args.i_img == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                img_dir = os.path.join(args.basedir, args.expname, 'img_spline_{:06d}'.format(i))
                os.makedirs(img_dir, exist_ok=True)
                for j, pose in enumerate(tqdm(spline_poses)):
                    # print(i, time.time() - t)
                    # t = time.time()
                    pose = pose[None, :3, :4]
                    ret = graph.render_video(i, pose[:3, :4], H, W, K, near, far, args)
                    rgbs = ret['rgb_map'].cpu().numpy()
                    rgb8 = to8b(rgbs)
                    imageio.imwrite(os.path.join(img_dir, 'rgb_{:03d}.png'.format(j)), rgb8)

                    depths = ret['disp_map'].cpu().numpy()
                    depths_ = depths / np.max(depths)
                    depth8 = to8b(depths_)
                    imageio.imwrite(os.path.join(img_dir, 'depth_{:03d}.png'.format(j)), depth8)

        if args.novel_view and i % args.i_novel_view == 0 and i > 0:
            # Turn on novel view testing mode
            i_ = torch.arange(0, images_all.shape[0], args.llffhold - 1)
            pose_test = poses_all[i_]
            poses_test_se3_ = SE3_to_se3_N(pose_test[:, :3, :4])
            model_test = novel_view_test.Model(poses_test_se3_, graph)
            graph_test = model_test.build_network(args)
            optimizer_test = model_test.setup_optimizer(args)
            for j in range(args.N_novel_view):
                ret_sharp, ray_idx_sharp, poses_sharp = graph_test.forward(i, depth_gts, img_idx, poses_num, H, W, K, near, far, args, novel_view=True)
                target_s_novel = images_novel.reshape(-1, H*W, 3)[:, ray_idx_sharp]
                target_s_novel = target_s_novel.reshape(-1, 3)
                loss_sharp = img2mse(ret_sharp['rgb_map'], target_s_novel)
                psnr_sharp = mse2psnr(loss_sharp)
                if 'rgb0' in ret_sharp:
                    img_loss0 = img2mse(ret_sharp['rgb0'], target_s_novel)
                    loss_sharp = loss_sharp + img_loss0
                if j%100==0:
                    print(psnr_sharp.item(), loss_sharp.item())
                optimizer_test.zero_grad()
                loss_sharp.backward()
                optimizer_test.step()
                decay_rate_sharp = 0.01
                decay_steps_sharp = args.lrate_decay * 100
                new_lrate_novel = args.pose_lrate * (decay_rate_sharp ** (j / decay_steps_sharp))
                for param_group in optimizer_test.param_groups:
                    if (j / decay_steps_sharp) <= 1.:
                        param_group['lr'] = new_lrate_novel * args.factor_pose_novel
            with torch.no_grad():
                imgs_render_novel = render_image_test(i, graph, poses_sharp, H, W, K, near, far, args, novel_view=3)

                mse_render = compute_img_metric(imgs_sharp_test, imgs_render_novel, 'mse')
                psnr_render = compute_img_metric(imgs_sharp_test, imgs_render_novel, 'psnr')
                ssim_render = compute_img_metric(imgs_sharp_test, imgs_render_novel, 'ssim')
                lpips_render = compute_img_metric(imgs_sharp_test, imgs_render_novel, 'lpips')

                with open(test_metric_file_novel, 'a') as outfile:
                    outfile.write(f"iter{i}: MSE3:{mse_render.item():.8f} PSNR3:{psnr_render.item():.8f}"
                                  f" SSIM3:{ssim_render.item():.8f} LPIPS3:{lpips_render.item():.8f}\n"
                                  )

                path_pose = os.path.join(basedir, expname, 'test_poses.txt')
                save_render_pose(poses_sharp, path_pose)

        if i % args.N_iters == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                path_pose = os.path.join(basedir, expname, 'train_poses.txt')
                i_render_pose = torch.arange(i_train.shape[0]) * args.deblur_images + args.deblur_images // 2
                render_poses_final = all_poses[i_render_pose]
                save_render_pose(render_poses_final, path_pose)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
