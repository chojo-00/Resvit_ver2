import os
import natsort
import pydicom
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from data.base_dataset import BaseDataset  # 수정: BaseDataset 임포트

EXTENSION = ['.png', '.tiff', '.dcm']


class DectDataset(BaseDataset):  # 수정: 클래스 이름 및 상속
    def initialize(self, opt):  # 수정: __init__ -> initialize, 파라미터 변경
        BaseDataset.initialize(self, opt)  # 수정: 부모 클래스 초기화
        self.mode = opt.phase  # 수정: 'mode'를 opt.phase에서 가져옴
        self.dataset_dir = os.path.join(opt.dataroot, self.mode)  # 수정: 경로 조합 방식 변경
        self.src = opt.src.split(',')  # 수정: opt에서 src 리스트 가져오기
        self.trg = opt.trg.split(',')  # 수정: opt에서 trg 리스트 가져오기
        
        if not os.path.exists(self.dataset_dir):
            raise IOError(f"Dataset path {self.dataset_dir} does not exist")

        """ Source """
        self.src_fnames = natsort.natsorted([os.path.relpath(os.path.join(root, fname), start=self.dataset_dir) for cls in self.src for root, _dirs, files in os.walk(os.path.join(self.dataset_dir, cls)) for fname in files])
        
        self.src_image_fnames = [
            fname for fname in self.src_fnames
            if self._file_ext(fname) in EXTENSION
            and os.path.getsize(os.path.join(self.dataset_dir, fname)) > 500
        ]
        print(f"[Dataset] After checking file extension and size, size={len(self.src_image_fnames)}!") # 수정: log.info -> print
        
        if len(self.src_image_fnames) == 0:
            raise IOError('No source image files found in the specified path')
        print(f"[Dataset] Built dataset {self.dataset_dir}, size={len(self.src_image_fnames)}!") # 수정: log.info -> print
        
        """ Target """
        self.trg_fnames = natsort.natsorted([os.path.relpath(os.path.join(root, fname), start=self.dataset_dir) for cls in self.trg for root, _dirs, files in os.walk(os.path.join(self.dataset_dir, cls)) for fname in files]) * len(self.src)
        
        self.trg_image_fnames = [
            fname for fname in self.trg_fnames
            if self._file_ext(fname) in EXTENSION
            and os.path.getsize(os.path.join(self.dataset_dir, fname)) > 500
        ]
        print(f"[Dataset] After checking file extension and size, size={len(self.trg_image_fnames)}!") # 수정: log.info -> print
        
        if len(self.trg_image_fnames) == 0:
            raise IOError('No target image files found in the specified path')
        print(f"[Dataset] Built dataset {self.dataset_dir}, size={len(self.trg_image_fnames)}!") # 수정: log.info -> print
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1,1]
        ])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @staticmethod
    def _is_valid_dcm(fpath):
        try:
            dcm = pydicom.dcmread(fpath, stop_before_pixels=False, force=True)
            return 'PixelData' in dcm
        except Exception as e:
            print(f"Error reading DICOM file {fpath}: {e}")
            return False
    
    def _open_file(self, fname):
        return open(os.path.join(self.dataset_dir, fname), 'rb')

    @staticmethod
    def _clip_and_normalize(img, min, max):
        img = np.clip(img, min, max)
        img = (img - min) / (max - min)
        return img
    
    def _CT_preprocess(self, dcm, img, window_width=None, window_level=None):
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        img = img * slope + intercept
        if window_width is not None and window_level is not None:
            min = window_level - (window_width / 2.0)
            max = window_level + (window_width / 2.0)
        else: # 12 bits
            min = -1024.0
            max = 3071.0
        img = self._clip_and_normalize(img, min, max)
        return img

    def __len__(self):
        return len(self.src_image_fnames)

    def __getitem__(self, index):
        src_fname = self.src_image_fnames[index]
        trg_fname = self.trg_image_fnames[index]
        src_fpath = os.path.join(self.dataset_dir, src_fname)
        trg_fpath = os.path.join(self.dataset_dir, trg_fname)

        with self._open_file(src_fname) as f:
            if self._file_ext(src_fname) == '.dcm':
                src_dcm = pydicom.dcmread(f, force=True)
                src_img = src_dcm.pixel_array.astype(np.float32)
                src_img = self._CT_preprocess(src_dcm, src_img, None, None)
            else: # jpg, jpeg, tiff, png, etc.
                src_img = np.array(Image.open(f)).astype(np.float32)

        with self._open_file(trg_fname) as f:
            if self._file_ext(trg_fname) == '.dcm':
                trg_dcm = pydicom.dcmread(f, force=True)
                trg_img = trg_dcm.pixel_array.astype(np.float32)
                trg_img = self._CT_preprocess(trg_dcm, trg_img, None, None)
            else: # jpg, jpeg, tiff, png, etc.
                trg_img = np.array(Image.open(f)).astype(np.float32)

        corrupt_img = self.transform(src_img[:,:,np.newaxis])
        clean_img = self.transform(trg_img[:,:,np.newaxis])

        # 수정: ResViT가 요구하는 dict 형태로 반환 ('A'가 입력, 'B'가 타겟)
        return {'A': corrupt_img, 'B': clean_img, 'A_paths': src_fpath, 'B_paths': trg_fpath}

    def name(self):
        return 'DectPairedDataset'