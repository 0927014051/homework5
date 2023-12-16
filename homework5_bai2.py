import numpy as np
import matplotlib.pyplot as plt

def read_data(file_path, size):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8, count=size*size)
        return np.reshape(data, (size, size))
X = read_data('dataset/girl2.sec', 256)
X1 = read_data('dataset/girl2Noise32.sec', 256)
X2 = read_data('dataset/girl2Noise32Hi.sec', 256)
Y1 = np.sum((X.astype("float") - X1.astype("float"))**2)
Y1 /= float(X.shape[0] * X1.shape[1])
Y2 = np.sum((X.astype("float") - X2.astype(float))**2)
Y2 /= float(X.shape[0] * X2.shape[1])
print("Mean Squared Error between girl2 and girl2Noise32 = ", Y1)
print("Mean Squared Error between girl2 and girl2Noise32Hi = ", Y2)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Image', fontsize=18)

plt.subplot(1, 3, 2)
plt.imshow(X1, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32 Image', fontsize=18)

plt.subplot(1, 3, 3)
plt.imshow(X2, cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32Hi Image', fontsize=18)
plt.show()

#Cau b
def stretch2(img):
    img_min = np.min(img)
    img_max = np.max(img)
    stretched_img = 255 * (img - img_min) / (img_max - img_min)
    return stretched_img.astype(np.uint8)

U_cutoff = 64
[U, V] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = np.double(np.sqrt(U**2 + V**2) <= U_cutoff)
HLtilde = np.fft.fftshift(HLtildeCenter)
Z = np.fft.ifft2(np.fft.fft2(X) * HLtilde)
Z1 = np.fft.ifft2(np.fft.fft2(X1) * HLtilde)
Z2 = np.fft.ifft2(np.fft.fft2(X2) * HLtilde)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(stretch2(Z), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('Original Image', fontsize=18)

plt.subplot(1, 3, 2)
plt.imshow(stretch2(Z1), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32 Image', fontsize=18)

plt.subplot(1, 3, 3)
plt.imshow(stretch2(Z2), cmap='gray', vmin=0, vmax=255)
plt.axis('image')
plt.axis('off')
plt.title('girl2Noise32Hi Image', fontsize=18)
plt.show()

Z3 = np.sum(np.abs(X.astype("complex") - Z.astype("complex")) ** 2)
Z3 /= float(X.shape[0] * Z.shape[1])
Z4 = np.sum(np.abs(X.astype("complex") - Z1.astype("complex")) ** 2)
Z4 /= float(X.shape[0] * Z1.shape[1])
Z5 = np.sum(np.abs(X.astype("complex") - Z2.astype("complex")) ** 2)
Z5 /= float(X.shape[0] * Z2.shape[1])
ISNR1 = 10*np.log10(Y1/Z4)
ISNR2 = 10*np.log10(Y2/Z5)
print('MSE of Z', Z3)
print('MSE of Z1', Z4)
print('MSE of Z2', Z5)
print('ISNR of girl2Noise32Hibin = ', ISNR2)
print('ISNR of girl2Noise32bin = ', ISNR1)

#Câu b vs Cau c
def GaussianLPF(U_cutoff_H, X, X1, X2):
    SigmaH = 0.19 * 256 / U_cutoff_H
    U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
    HtildeCenter = np.exp((-2 * np.pi ** 2 * SigmaH ** 2) / (256 ** 2) * (U ** 2 + V ** 2))
    Htilde = np.fft.fftshift(HtildeCenter)
    H = np.fft.ifft2(Htilde)
    H2 = np.fft.fftshift(H)
    ZPH2 = np.zeros((512, 512))
    ZPH2[:256, :256] = H2
    ZPX = np.zeros((512, 512))
    ZPX[:256, :256] = X
    yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
    T = yy[128:384, 128:384]
    ZPX[:256, :256] = X1
    yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
    T1 = yy[128:384, 128:384]
    ZPX[:256, :256] = X2
    yy = np.fft.ifft2(np.fft.fft2(ZPX) * np.fft.fft2(ZPH2))
    T2 = yy[128:384, 128:384]
    ET = np.sum(np.abs(X.astype("complex") - T.astype("complex")) ** 2)
    ET /= float(X.shape[0] * T.shape[1])
    ET1 = np.sum(np.abs(X.astype("complex") - T1.astype("complex")) ** 2)
    ET1 /= float(X.shape[0] * T1.shape[1])
    ET2 = np.sum(np.abs(X.astype("complex") - T.astype("complex")) ** 2)
    ET2 /= float(X.shape[0] * T2.shape[1])
    ISNR3 = 10 * np.log10(Y1 / ET1)
    ISNR4 = 10 * np.log10(Y2 / ET2)
    print('MSE of girl2bin', ET)
    print('MSE of girl2Noise32bin', ET1)
    print('MSE of girl2Noise32Hibin', ET2)
    print('ISNR of girl2Noise32Hibin = ', ISNR3)
    print('ISNR of girl2Noise32bin = ', ISNR4)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(stretch2(T), cmap='gray', vmin=0, vmax=255)
    plt.axis('image')
    plt.axis('off')
    plt.title('Original Image', fontsize=18)

    plt.subplot(1, 3, 2)
    plt.imshow(stretch2(T1), cmap='gray', vmin=0, vmax=255)
    plt.axis('image')
    plt.axis('off')
    plt.title('girl2Noise32 Image', fontsize=18)

    plt.subplot(1, 3, 3)
    plt.imshow(stretch2(T2), cmap='gray', vmin=0, vmax=255)
    plt.axis('image')
    plt.axis('off')
    plt.title('girl2Noise32Hi Image', fontsize=18)
    plt.show()
GaussianLPF(64, X, X1, X2)
GaussianLPF(77.5, X, X1, X2)














