import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import wandb

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # =========================================================
    # [추가] wandb 초기화
    # =========================================================
    wandb_run = wandb.init(
        project="ResViT-DECT",
        name=opt.name,
        config=vars(opt),
        resume="allow",
    )
    print(f"[wandb] Run URL: {wandb_run.url}")

    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()

    # =========================================================
    # [수정] validation data — 폴더 존재 시에만 로드
    # =========================================================
    val_dir = os.path.join(opt.dataroot, 'val')
    use_val = os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0
    if use_val:
        opt.phase='val'
        data_loader_val = CreateDataLoader(opt)
        dataset_val = data_loader_val.load_data()
        dataset_size_val = len(data_loader_val)
        print('#Validation images = %d' % dataset_size_val)
        if opt.model=='cycle_gan':
            L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
            psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])            
        else:
            L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
            psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])
        opt.phase='train'  # phase 복원
    else:
        print('[Info] No validation folder found at %s, skipping validation.' % val_dir)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    
    print("start training")

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                else:
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    


            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

                # [추가] wandb 학습 loss 로깅
                wandb_log = {"epoch": epoch, "iter": total_steps}
                for k, v in errors.items():
                    wandb_log[f"train/{k}"] = v
                wandb_log["train/lr"] = model.optimizers[0].param_groups[0]['lr']
                wandb.log(wandb_log, step=total_steps)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        #Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            # =========================================================
            # [수정] validation — use_val일 때만 실행
            # =========================================================
            if use_val:
                logger = open(os.path.join(save_dir, 'log.txt'), 'a')
                print(opt.dataset_mode)
                opt.phase='val'
                for i, data_val in enumerate(dataset_val):
                    model.set_input(data_val)
                    model.test()
                    fake_im=model.fake_B.cpu().data.numpy()
                    real_im=model.real_B.cpu().data.numpy() 
                    real_im=real_im*0.5+0.5
                    fake_im=fake_im*0.5+0.5
                    if real_im.max() <= 0:
                        continue
                    L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                    psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max())

                l1_avg_loss = np.mean(L1_avg[epoch-1])
                mean_psnr = np.mean(psnr_avg[epoch-1])
                std_psnr = np.std(psnr_avg[epoch-1])
                print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f ' % \
                (epoch, l1_avg_loss, mean_psnr,std_psnr))
                print_log(logger,'')
                logger.close()

                # [추가] wandb validation 로깅
                wandb.log({
                    "val/l1_avg": l1_avg_loss,
                    "val/psnr_mean": mean_psnr,
                    "val/psnr_std": std_psnr,
                    "epoch": epoch,
                }, step=total_steps)

            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
            model.save('latest')
            model.save(epoch)

            # [추가] wandb 생성 이미지 샘플 로깅
            visuals = model.get_current_visuals()
            wandb.log({
                "samples/input": wandb.Image(visuals['real_A'], caption="Input (src keV)"),
                "samples/generated": wandb.Image(visuals['fake_B'], caption="Generated (70keV)"),
                "samples/ground_truth": wandb.Image(visuals['real_B'], caption="GT (70keV)"),
                "epoch": epoch,
            }, step=total_steps)

        epoch_time = time.time() - epoch_start_time
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, epoch_time))

        # [추가] wandb epoch 시간 로깅
        wandb.log({"epoch_time_sec": epoch_time, "epoch": epoch}, step=total_steps)

        model.update_learning_rate()

    # [추가] wandb 종료
    wandb.finish()