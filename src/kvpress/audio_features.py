import numpy as np
import scipy.signal

def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """创建与 SpeechLib FbankFC 相同的梅尔滤波器组。
    
    Args:
        sample_rate (int): 采样率 (Hz)
        n_fft (int): FFT 大小
        n_mels (int): 梅尔滤波器大小
        fmin (float): 最低频率 (Hz)
        fmax (float): 最高频率 (Hz)
    
    Returns:
        np.ndarray: 梅尔变换矩阵 [shape=(n_mels, 1 + n_fft/2)]
    """
    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negative"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)
    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class AudioFeatureExtractor:
    """简化版音频特征提取器，基于 NumPy，不依赖外部库"""
    
    def __init__(self, feature_size=80, sampling_rate=16000):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        
        # 初始化 Mel 滤波器
        self._mel = speechlib_mel(sampling_rate, 512, feature_size, fmin=None, fmax=7690).T
        
        # 初始化汉明窗
        self._hamming400 = np.hamming(400)  # 用于 16kHz 音频
        self._hamming200 = np.hamming(200)  # 用于 8kHz 音频
    
    def _extract_spectrogram(self, wav, fs):
        """从波形提取频谱图特征
        
        Args:
            wav (1D array): 输入波形
            fs (int): 采样率，16000 或 8000
                如果 fs=8000，波形将被重采样到 16000Hz
        
        Returns:
            2D array: 频谱图特征矩阵
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)

        # 默认提取立体声的均值
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # 根据需要重采样到 16000 或 8000
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"不支持的采样率 {fs}")

        # 处理 8kHz 音频
        if fs == 8000:
            # 不进行重采样，直接填充零
            pass
        elif fs != 16000:
            raise RuntimeError(f"输入数据使用不支持的采样率: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: 截断不足一个 hop 的剩余样本
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        
        # 创建帧
        y_frames = np.array(
            [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )

        # Spec 2: 在每个批次中应用预加重
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        # 计算 FFT
        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)

        if fs == 8000:
            # 需要进行填充以模拟 16kHz 数据，但 4-8kHz 频段以零填充
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)  # Nyquist bin 设为零

        spec = np.abs(S).astype(np.float32)
        return spec

    def extract_features(self, wav, fs):
        """从波形提取对数滤波器组特征
        
        Args:
            wav (1D array): 输入波形
            fs (int): 采样率，16000 或 8000
        
        Returns:
            2D array: 对数梅尔滤波器组特征矩阵，形状为 (T, 80)
        """
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank

    def __call__(self, audio_data, sampling_rate=16000):
        """提取音频特征
        
        Args:
            audio_data: 音频数据
            sampling_rate: 采样率
            
        Returns:
            np.ndarray: 提取的特征
        """
        return self.extract_features(audio_data, sampling_rate)