# Reference from VideoMAE
import os
import numpy as np
import torch
import decord
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.functional import amplitude_to_DB
from torchvision import transforms
import math
import copy
import librosa.display
class VoxCeleb2Dataset(BaseDataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """
    # def __init__(self,
    #              root,
    #              setting,
    #              train=True,
    #              test_mode=False,
    #              video_ext='mp4',
    #              is_color=True,
    #              num_segments=1,
    #              num_crop=1,
    #              new_length=1,
    #              new_step=1,
    #              transform=None,
    #              temporal_jitter=False,
    #              video_loader=False,
    #              use_decord=False,
    #              lazy_init=False):
    def __init__(self, opt, transform=None):

        BaseDataset.__init__(self, opt)
        self.root = opt.dataroot
        self.setting = opt.setting
        self.num_segments = 1
        self.new_length = opt.num_frames
        self.new_step = opt.sampling_rate
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = False
        self.video_loader = True
        self.video_ext = 'mp4'
        self.lazy_init = False
        self.video_transform = transform

        self.video_dir = os.path.join(self.root, "video")
        if not self.lazy_init:
            self.clips = self._make_dataset(self.root, self.setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))
    
    def __getitem__(self, index):
        directory, target, duration = copy.deepcopy(self.clips[index])
        directory = os.path.join(self.video_dir, directory)
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=3)
            fps = int(decord_vr.get_avg_fps())
            decord_ar = decord.AudioReader(video_name)
            # decord.bridge.set_bridge('torch')
            # decord_vr = decord.AVReader(video_name, sample_rate=16000, ctx=decord.cpu(0))
            # fps = 25
            # audio, video = decord_vr.get_batch(range(self.skip_length))
            # decord_ar = [a.asnumpy() for a in audio]
            
            

        # skip some frames of the video
        segment_indices, skip_offsets = self._sample_train_indices(int(duration))
        images, frame_id_list = self._video_TSN_decord_batch_loader(directory, decord_vr, int(duration), segment_indices, skip_offsets)
        audio = self._audio_transform(decord_ar, frame_id_list, fps)    # (1, T, F)

        process_data, mask = self.video_transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask, audio)
        # return (audio, audio, audio)


    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        setting = os.path.join(directory, setting)
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                duration = int(line_info[2])
                if duration < self.new_length:
                    # 过滤持续时间少于所需帧数的视频
                    continue
                item = (clip_path, target, duration)
                clips.append(item)
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            max_frame = average_duration - self.new_length if average_duration - self.new_length > 0 else average_duration
            offsets = offsets + np.random.randint(max_frame,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets


    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    # frame_id = offset - 1
                    break
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            if duration < self.skip_length:
                video_data = video_reader.get_batch(list(range(duration))).asnumpy()
                # audio, video_data = video_reader.get_batch(list(range(duration)))
                # video_data = video_data.asnumpy()
                padding = np.zeros((self.skip_length - len(video_data), video_data.shape[1], video_data.shape[2], video_data.shape[3]))
                video_data = np.vstack((video_data, padding)) #在末尾补齐空图像
                for vid, _ in enumerate(range(self.skip_length)):
                    print(video_data[vid, :,:,:].shape)
                sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(range(self.skip_length))]
            else:
                video_data = video_reader.get_batch(frame_id_list).asnumpy()
                # audio, video_data = video_reader.get_batch(frame_id_list)
                # video_data = video_data.asnumpy()
                sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        
        # audio = [a.asnumpy() for a in audio]
        # return np.array(sampled_list), audio, frame_id_list
        return sampled_list, frame_id_list

    
    def _audio_transform(self, decord_ar, frame_id_list, fps):
        sample_rate = 16000
        hop_length = 160
        num = sample_rate // fps # 每帧对应的音频特征数
        audio_feature_id_list = np.multiply(frame_id_list, num) # 视频帧对应的音频特征

        audio_transform = transforms.Compose([
            MelSpectrogram(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_fft=512,
                n_mels=128
            ),
            # Lambda(
            #     lambda x: x[:, :-(x.size(1) - args.dataset.fps * args.audio2video)]
            #     if x.size(1) > args.dataset.fps * args.audio2video else x
            # ),  # 截断长度为1秒
            transforms.Lambda(lambda x: amplitude_to_DB(
                x, multiplier=10, amin=1e-10, db_multiplier=math.log10(max(1e-10, torch.max(x).item())), top_db=80
            )),
            transforms.Lambda(lambda x: (x + 40) / 40),
            # transforms.Lambda(lambda x: x.transpose(1, 0).unsqueeze(0)),  # (F, T) -> (1, T, F)
            transforms.Lambda(lambda x: x.transpose(1, 2)),  # (1, F, T) -> (1, T, F)

        ])
        audio = copy.deepcopy(decord_ar._array)
        # audio = torch.tensor(decord_ar,dtype=torch.float32)
        # audio sample according to video
        audio_sample_list = np.array([])
        if self.new_step == 1: # no skip
            audio_sample_list = audio[:,audio_feature_id_list[0]: audio_feature_id_list[-1] + num].copy()
        else:
            for i in audio_feature_id_list:
                audio_sample_list.append(audio[i : i + num])
            
        audio_sample_list = torch.tensor(audio_sample_list, dtype=torch.float32)
        # 补齐为指定frames对应的音频长度
        if audio_sample_list.shape[1] < self.new_length * num:
            audio_sample_list = torch.nn.functional.pad(audio_sample_list, (0, self.new_length * num - audio_sample_list.shape[1]))
        # if audio_sample_list.size(0) < self.new_length * num:
        #     audio_sample_list = torch.nn.functional.pad(audio_sample_list, (0, self.new_length * num - audio_sample_list.size(0)))

        return audio_transform(audio_sample_list)
        # return audio_transform(audio)

class myMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=16000, hop_length=256, n_fft=1024, n_mels=24, win_length=None):
        super().__init__()
        
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.mel_basis = torch.from_numpy(self.mel_basis).to(torch.float32)
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = torch.hann_window(self.win_length)
    def forward(self, x):
        stft = torch.stft(x,
                          win_length=self.win_length,
                          hop_length=self.hop_length,
                          n_fft=self.n_fft,
                          window=self.window)
        # real = stft[:, :, :, 0]
        # im = stft[:, :, :, 1]
        # spec = torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))
        spec = torch.norm(stft, p=2, dim=-1).to(torch.float32)
        # convert linear spec to mel
        mel = torch.matmul(self.mel_basis,spec)

        # mfcc = scipy.fftpack.dct(mel, axis=0, type=4, norm="ortho")[:20]
        return mel 