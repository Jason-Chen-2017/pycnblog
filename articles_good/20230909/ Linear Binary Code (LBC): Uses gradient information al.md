
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、产品概述
Linear Binary Code (LBC)是一种编码方式，其特点在于每一个二进制码都由一条线性函数所组成，也就是说，一条线上的每个像素都对应着该线上两个特征点的像素值之间的差异，而且这些差异是通过直线方程计算得出的，因此LBC具有良好的分辨率和灵活性。LBC主要用于图像去噪和特征提取等领域。
## 二、产品优势
### （一）分辨率高
由于LBC每条线都是直线，因此它能够保持较高的分辨率。对于一些密集场景中的图像来说，采用LBC可以有效地降低噪声，并且还能保持原图中的细节信息。
### （二）灵活性强
因为LBC将图像看作由线段构成的曲线，因此在提取图像特征时，它能够考虑到相邻像素间的位置关系。这种方法很适合处理图像的光照不均匀或非线性结构变换的场景。
### （三）自适应性强
LBC具有自适应性强，不仅仅可以在任何一种场景下使用，而且可以根据数据自身的特性进行自学习。由于LBC是直线上的曲线，因此对噪声和摩尔效应非常敏感。在一般情况下，LBC也能有效地消除图像中高频噪声。
### （四）计算效率高
由于LBC基于曲线拟合的方法，所以它的计算效率高，在很多情况下都可以达到实时的要求。
### （五）无损压缩能力强
LBC由于采用了线性变换的方式，因此其无损压缩能力很强，在某些情况下甚至可以达到无损的效果。
## 三、产品架构

如图1所示，LBC的架构分为前端模块、编码器和后端模块三个部分，它们之间相互配合，完成对图像的处理流程。

* 前端模块负责图像预处理阶段，包括图像降采样、边缘检测、直方图均衡化等工作，确保图像质量可靠并得到清晰的输入；
* 编码器则接收前端模块输出的图像，按照线性二进制码（LBC）的方式进行处理，得到一个连续序列的码流，其中包括多个连续坐标轴及其值。编码器首先对图像的像素值进行离散化，然后求取图像的梯度信息，并利用梯度信息计算出每一个像素值对应着该线上两个特征点的像素值的差异；
* 后端模块接收编码器输出的码流，并对其进行解码，从而恢复出原始图像。后端模块首先将码流进行重建，将连续坐标轴及其值转换回对应的图像，然后再用这些值进行反向映射，恢复出原始图像。

# 2.核心概念术语
## 2.1 LBC编码过程
首先要对图像进行预处理，包括图像降采样、边缘检测、直方图均衡化等工作，确保图像质量可靠并得到清晰的输入。

之后，把每个像素的灰度值映射到[0, N]区间，N为码元个数。例如，若N=8，则像素值范围为[0, 255]映射到[0, 7]的区间。

然后，对每个坐标轴建立一个等距采样序列，形成等距像素矩阵。例如，假设图像大小为$W\times H$，则对于垂直方向，各行的采样点距离为$\frac{H}{2}$，对于水平方向，各列的采样点距离为$\frac{W}{2}$。

这样，就获得了一个$N\times M$维的矩阵，其中M为矩阵行数，N为矩阵列数。对于每个坐标轴上的$m$个点$(x_i, y_j)$，计算$p_{ij}=I(x_i,y_j)-I(x_{i-1},y_{j})$，即第$i$行、第$j$列对应的像素值减去第$i-1$行、第$j$列对应的像素值，作为对应位置的差值。

注意：这里需要注意，这里不是简单的用差值作为差值矩阵的元素，而是用差值除以$\sqrt{(x_i-x_{i-1})(y_j-y_{j-1})}$(权重系数)作为差值矩阵的元素，目的是为了消除不同距离下的差值影响。这样做的原因是，不同的距离下的差值对计算结果的贡献是不同的，如果直接用差值作为差值矩阵的元素，会使得不同距离下的差值过于平滑。而采用权重系数的方法，可以让距离大的差值占主导作用，而距离小的差值对结果的贡献可以忽略不计。

然后，计算梯度信息，即求取图像的梯度方向和长度。首先，计算二阶微分矩阵。对于坐标轴上的$m$个点$(x_i, y_j)$，分别计算$(\partial^2 I / \partial x^2)(x_i,y_j), (\partial^2 I / \partial y^2)(x_i,y_j), (\partial^2 I / \partial xy)(x_i,y_j)$。

其次，求取梯度方向和长度。取二阶微分矩阵的特征值，得到最长的特征值对应的方向，其方向向量为单位方向向量$u=(\cos{\theta},\sin{\theta})$，特征值为$\lambda_{\max}$。

最后，使用梯度信息构建等距曲线。对于图像中$k$条线段，通过梯度方向和长度计算得到一系列的控制点$(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$，表示一条等距曲线。把每个控制点映射到区间[0, N]上，得到相应的二进制码。

## 2.2 LBC解码过程
首先，读取每条线段对应的控制点，确定每条线段的方向向量和长度。

其次，将坐标轴上的值从[-N, N]映射到[0, 1]区间。例如，对于垂直方向，除以$N$，对于水平方向，除以$2N$。

然后，根据每条线段的方向向量和长度计算插值矩阵。对于坐标轴上的$m$个点$(x_i, y_j)$，确定一条射线$(x_i,\tilde{y}_i),(x_{i+1},\tilde{y}_{i+1}),\cdots,(x_j,\tilde{y}_j)$，其中$\tilde{y}_i=\pm(\cos{\theta}x_i+\sin{\theta}y_i)+y_i$。这条射线沿着坐标轴直线段的方向，穿过控制点$(x_i,y_i)$和$(x_{i+1},y_{i+1})$。

对于每条射线上的$s$值，计算映射点$(\tilde{x}_i,\tilde{y}_i+sy)$对应的图像值$I(\tilde{x}_i,\tilde{y}_i+sy)$。

最后，把$M$个列向量按顺序拼接起来，就是最终的图像矩阵。
# 3.具体算法原理
## 3.1 梯度信息的计算
首先，把图像的每个像素值映射到[0, N]区间，N为码元个数。例如，若N=8，则像素值范围为[0, 255]映射到[0, 7]的区间。

然后，对每个坐标轴建立一个等距采样序列，形成等距像素矩阵。例如，假设图像大小为$W\times H$，则对于垂直方向，各行的采样点距离为$\frac{H}{2}$，对于水平方向，各列的采样点距离为$\frac{W}{2}$。

这样，就获得了一个$N\times M$维的矩阵，其中M为矩阵行数，N为矩阵列数。对于每个坐标轴上的$m$个点$(x_i, y_j)$，计算$p_{ij}=I(x_i,y_j)-I(x_{i-1},y_{j})$，即第$i$行、第$j$列对应的像素值减去第$i-1$行、第$j$列对应的像素值，作为对应位置的差值。

注意：这里需要注意，这里不是简单的用差值作为差值矩阵的元素，而是用差值除以$\sqrt{(x_i-x_{i-1})(y_j-y_{j-1})}$(权重系数)作为差值矩阵的元素，目的是为了消除不同距离下的差值影响。这样做的原因是，不同的距离下的差值对计算结果的贡献是不同的，如果直接用差值作为差值矩阵的元素，会使得不同距离下的差值过于平滑。而采用权重系数的方法，可以让距离大的差值占主导作用，而距离小的差值对结果的贡献可以忽略不计。

然后，计算梯度信息，即求取图像的梯度方向和长度。首先，计算二阶微分矩阵。对于坐标轴上的$m$个点$(x_i, y_j)$，分别计算$(\partial^2 I / \partial x^2)(x_i,y_j), (\partial^2 I / \partial y^2)(x_i,y_j), (\partial^2 I / \partial xy)(x_i,y_j)$。

其次，求取梯度方向和长度。取二阶微分矩阵的特征值，得到最长的特征值对应的方向，其方向向量为单位方向向量$u=(\cos{\theta},\sin{\theta})$，特征值为$\lambda_{\max}$。

最后，使用梯度信息构建等距曲线。对于图像中$k$条线段，通过梯度方向和长度计算得到一系列的控制点$(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$，表示一条等距曲线。把每个控制点映射到区间[0, N]上，得到相应的二进制码。

## 3.2 LBC编码过程
首先，对图像进行预处理，包括图像降采样、边缘检测、直方图均衡化等工作，确保图像质量可靠并得到清晰的输入。

之后，把每个像素的灰度值映射到[0, N]区间，N为码元个数。例如，若N=8，则像素值范围为[0, 255]映射到[0, 7]的区间。

然后，对每个坐标轴建立一个等距采样序列，形成等距像素矩阵。例如，假设图像大小为$W\times H$，则对于垂直方向，各行的采样点距离为$\frac{H}{2}$，对于水平方向，各列的采样点距离为$\frac{W}{2}$。

这样，就获得了一个$N\times M$维的矩阵，其中M为矩阵行数，N为矩阵列数。对于每个坐标轴上的$m$个点$(x_i, y_j)$，计算$p_{ij}=I(x_i,y_j)-I(x_{i-1},y_{j})$，即第$i$行、第$j$列对应的像素值减去第$i-1$行、第$j$列对应的像素值，作为对应位置的差值。

注意：这里需要注意，这里不是简单的用差值作为差值矩阵的元素，而是用差值除以$\sqrt{(x_i-x_{i-1})(y_j-y_{j-1})}$(权重系数)作为差值矩阵的元素，目的是为了消除不同距离下的差值影响。这样做的原因是，不同的距离下的差值对计算结果的贡献是不同的，如果直接用差值作为差值矩阵的元素，会使得不同距离下的差值过于平滑。而采用权重系数的方法，可以让距离大的差值占主导作用，而距离小的差值对结果的贡献可以忽略不计。

然后，计算梯度信息，即求取图像的梯度方向和长度。首先，计算二阶微分矩阵。对于坐标轴上的$m$个点$(x_i, y_j)$，分别计算$(\partial^2 I / \partial x^2)(x_i,y_j), (\partial^2 I / \partial y^2)(x_i,y_j), (\partial^2 I / \partial xy)(x_i,y_j)$。

其次，求取梯度方向和长度。取二阶微分矩阵的特征值，得到最长的特征值对应的方向，其方向向量为单位方向向量$u=(\cos{\theta},\sin{\theta})$，特征值为$\lambda_{\max}$。

最后，使用梯度信息构建等距曲线。对于图像中$k$条线段，通过梯度方向和长度计算得到一系列的控制点$(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$，表示一条等距曲线。把每个控制点映射到区间[0, N]上，得到相应的二进制码。
# 4.具体代码实例
## 4.1 Matlab示例代码
```matlab
% example data: a line with salt and pepper noise
img = imread('line_noise.bmp'); % load image from file
img = rgb2gray(img);            % convert to gray scale
sz = size(img);                 % get the size of image

% parameters for LBC encoding
N = 8;                          % number of binary code elements
wsize = [9 9];                  % window size
step = ceil([3 3]/2)*2 + 1;     % step size
precision = 'double';           % precision of floating point numbers

% preprocess image by denoising and edge detection
img = denoise_tv_bregman(img, wsize, 1e-1, 1);   % denoising using total variation filter
img = medfilt2(img, [3 3]);                     % median filtering
edges = canny(img, 0.5, 1);                      % detect edges
img = img*(1-edges);                            % remove edges in image

% calculate difference matrix and gradient direction and length
diffmat = double((imgrad(img)));                % calculate gradient values at each pixel position
weightmat = sqrt((diffmat(:,2).^2+diffmat(:,1).^2))./step;    % weight matrix based on distance between neighboring pixels
weightmat = round(weightmat * N)./ N;                    % quantize weights into integer range of [0, N]
encmatrix = zeros(length(diffmat), 2);               % initialize encoding matrix
for i=2:sz(1)
    for j=2:sz(2)
        dx = diffmat(j,:)';                         % extract first order derivative components
        dy = diffmat(j+1,:)';                        % use symmetry property of second order derivatives
        gval = sqrt(dx'*dx+dy'*dy)/sqrt(step^2);      % calculate gradient magnitude and normalize it
        if isnan(gval)
            continue                                % skip nan gradients
        end
        theta = atan2(dy', dx');                   % calculate angle of gradient vector
        u = [cos(theta)', sin(theta)');              % unit vector of gradient direction
        idx = find(diffmat(j,:)==inf)';             % handle infinity values due to sampling errors
        while ~isempty(idx)                           % replace infinity values with finite approximation
            infloc = idx(randi(length(idx), 1));        % choose one random index that contains an infinite value
            [gradx, grady] = ndgrid(-step/(2*step), -step/(2*step)):step/(2*step):step/(2*step);
            gradx(:) = cosd(theta).* gradx(:) - sind(theta).* grady(:);
            grady(:) = sind(theta).* gradx(:) + cosd(theta).* grady(:);
            [gx, gy] = real(ifft2(fft2(diffmat(j,1))+fft2(diffmat(j+1,1))*exp(-1i*gradx.*i)+(gradx.*grady.*diffmat(j+1,2))));
            gx = mean(gx(:))/norm([step step], 2);          % approximate gradient values as average of surrounding four points
            gy = mean(gy(:))/norm([step step], 2);
            diffmat(j+(infloc-1)*step,:) = [[gx gy]];       % fill infinity values with approximated values
            idx = find(diffmat(j,:)==inf)';             % update list of indices containing infinity values
        end
        encmatrix(j-1+i*(sz(2)-1),:) = [(floor((dx')*weightmat)' + floor((dy')*weightmat)')'. (round(gval*N))]';  % encode gradient info
    end
end

% perform adaptive clustering on encoded gradient vectors
K = max(2, min(32, sum(isfinite(encmatrix))))         % determine number of clusters based on non-nan values in encoding matrix
data = encmatrix(isfinite(encmatrix));                  % select only valid data points for clustering
model = kmeans(data, K);                                  % train clustering model
code = uint8(zeros(ceil(sz(1)/(step)), ceil(sz(2)/(step))).');         % initialize empty binary code matrix
for i=1:(ceil(sz(1)/(step)))
    for j=1:(ceil(sz(2)/(step)))
        p = model.centroids(model.assignments==uint8(((i-1)*(sz(2)-(step+1))+j)::((i-1)*(sz(2)-(step+1))+j+step^2-1)));   % assign centroids according to cluster assignment
        dists = sum((data-repmat(p, size(data, 1), 1)).^2, 2).^(1/2);   % compute distances between data and assigned centroids
        code((i-1)*step+1::(i-1)*step+step, (j-1)*step+1::(j-1)*step+step) = mod(argmin(dists)+1, N+1);  % map data points to closest centroid
    end
end

% write binary codes to output file
fid = fopen('output.bin','w');                               % open output file
fwrite(fid, float(code), precision);                          % save binary code as floats
fclose(fid);                                                 % close output file
```
## 4.2 Python示例代码
```python
import numpy as np
from skimage import io, color, filters, feature, morphology, exposure


def lbc_encode(img, nbits, window_shape, step_shape):

    # Grayscale conversion
    img = color.rgb2gray(img)

    # Denoising
    img = filters.denoise_tv_chambolle(img, weight=1.0, multichannel=False)
    
    # Median filtering
    img = filters.median(img, selem=np.ones(window_shape))

    # Edge detection
    mask = feature.canny(img, sigma=0.5)
    img *= 1 - mask

    # Calculate gradient
    fx, fy = np.gradient(img)
    g = np.hypot(fx, fy)
    th = np.arctan2(fy, fx)

    # Quantize gradient strength
    q = np.round(nbits * g / g.max())
    q[q > nbits] = nbits

    # Compute gradient orientation distribution
    bins = int(np.pi / (2 * np.arcsin(1 / nbits) ** 2))
    hist, _ = np.histogram(th, bins=bins, range=[0, 2 * np.pi])
    phist = np.cumsum(hist) / len(th)

    # Define mapping function
    def mapping_func(t):
        return np.interp(t, phist[:-1], np.arange(nbits))

    # Encode histogram using curve fitting
    t = np.linspace(phist[0], phist[-1], num=len(phist) * 2)
    c = np.polyfit(phist, mapping_func(phist), deg=1)[::-1]
    h = np.clip(mapping_func(t), 0, nbits - 1)

    # Split the input image into small blocks
    block_size = tuple(s // ss for s, ss in zip(img.shape, step_shape))
    grid = [(slice(ss * i, None if s == sb else ss * (i + 1))
             for i, s, sb in zip(range(ss), img.shape, block_size))
            for ss in step_shape]
    steps = np.array(block_size) // 2

    # Allocate memory for coding result
    coded_blocks = np.empty((*steps, nbits), dtype='int32')

    # Iterate over all blocks
    for block in grid:

        # Extract subimage centered around current block
        patch = img[tuple(slice(s - bs // 2, None if bs == b else s + bs // 2)
                          for s, bs, b in zip(block.indices(img.shape),
                                              block_size, img.shape))]
        
        # Calculate differences
        dxs = patch[:, :-1] - patch[:, 1:]
        dys = patch[:-1, :] - patch[1:, :]

        # Quantize differences
        wx = np.mean(np.abs(dxs), axis=-1) < 0.01
        wy = np.mean(np.abs(dys), axis=-1) < 0.01
        dx = np.round(nbits * dxs[..., ~wx] / abs(dxs[..., ~wx]).max()).astype('int32')
        dy = np.round(nbits * dys[..., ~wy] / abs(dys[..., ~wy]).max()).astype('int32')
        
        # Concatenate deltas and gradient magnitude
        delta = np.stack([dx, dy], axis=-1)
        mag = np.round(nbits * g[block].reshape((-1,)) / g[block].max()).reshape((-1,))
        vect = np.concatenate([delta, mag.reshape((-1, 1))], axis=-1)

        # Map to discrete colors
        colormap = mapping_func(phist[:nbits]).reshape((-1,))
        cols = colormap[vect]

        # Assign columns to nearest segment center
        ind = ((cols // step_shape[1]) * step_shape[1]
               + (cols % step_shape[1])) + (block[0][0] // step_shape[0]) * step_shape[0] * step_shape[1]
        ind += np.arange(bs[0]*bs[1])
        unique, counts = np.unique(ind, return_counts=True)
        segments = dict(zip(unique, counts))
        labels = {v: k for k, v in enumerate(sorted(segments))}
        bincodes = np.vectorize(labels.__getitem__)(ind)
        coded_blocks[(steps * (block[0][:2] // step_shape[:2])).tolist()] = bincodes

    # Merge blocks back to image
    decoded_img = np.zeros((*img.shape, nbits), dtype='bool')
    decoded_img[steps:-steps, steps:-steps] = masked_select_segments(coded_blocks)
    
    # Inverse coding process
    inverted_img = np.zeros_like(decoded_img)
    invmap = lambda x: np.interp(x, np.arange(nbits), colormap)
    for r in range(inverted_img.shape[0]):
        for c in range(inverted_img.shape[1]):

            # Extract corresponding column and decode color
            codes = decoded_img[r, c]
            cols = invert_mapping_func(invmap(codes))[0]
            
            # Dequantize differences
            dx = cols[..., :2] * (-2**(nbits-1) if any(cols[..., :2]<0) else 2**(nbits-1-cols[..., :2]))
            dy = cols[..., 2] * (-2**(nbits-1) if any(cols[..., 2]<0) else 2**(nbits-1-cols[..., 2]))

            # Interpolate missing positions
            bx = [-dx[:, :, 0], -dx[:, :, 1]]
            by = [-dy[:, :, 0], -dy[:, :, 1]]
            b = bx + by
            A = np.zeros((*bx[0].shape, 2, 2))
            b = np.ravel(b, order='F').T
            for i in range(len(A)):
                A[i, :, 0] = bx[0][i,:]
                A[i, :, 1] = bx[1][i,:]
            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            for i in range(len(coeff)):
                bx[0][i,:] -= coeff[i,0]*by[0][i,:] - coeff[i,1]*by[1][i,:]
                bx[1][i,:] -= coeff[i,0]*by[1][i,:] + coeff[i,1]*by[0][i,:]
            bx[0] /= np.linalg.norm(bx[0], ord=2, axis=-1).reshape((-1,1))
            bx[1] /= np.linalg.norm(bx[1], ord=2, axis=-1).reshape((-1,1))
            dx[:, :, 0] = bx[0]+dx[:, :, 0]
            dx[:, :, 1] = bx[1]+dx[:, :, 1]
            dy[:, :, 0] = bx[0]-dy[:, :, 0]
            dy[:, :, 1] = bx[1]-dy[:, :, 1]
            dxs = dx[:, :, 0]+dx[:, :, 1]
            dys = dy[:, :, 0]+dy[:, :, 1]

            # Substitute patches
            bw = block_size[0]//2
            bh = block_size[1]//2
            src = np.zeros((*block_size, 2))
            dst = np.zeros_like(src)
            src[..., 0] = np.arange(-bw, bw+1)
            src[..., 1] = 0
            dst[..., 0] = -(step_shape[1]//2) + bx[:,:,0]
            dst[..., 1] = 0
            src[..., 1] = np.arange(-bh, bh+1)
            dst[..., 1] = -(step_shape[0]//2) + bx[:,:,1]
            warped_dxs = interpolate_patch(src, dst, dxs)
            warped_dys = interpolate_patch(src, dst, dys)
            warped_mag = interpolate_patch(src, dst, mag.reshape((-1, 1)))
            decoded_img[r, c] = (warped_dxs<0)&(warped_dys>0)&(~np.isnan(warped_dxs))&(~np.isnan(warped_dys))&\
                                (~np.isnan(warped_mag))&mask[r,c]
            
    return decoded_img
    

def lbc_decode(img, nbits, window_shape, step_shape):

    # Retrieve shape information from input image
    height, width = img.shape[:2]
    pad_height = (width // step_shape[1]) * step_shape[1] + 1
    pad_width = (height // step_shape[0]) * step_shape[0] + 1
    padded_img = np.pad(img, ((0, pad_height-height), (0, pad_width-width)), mode='constant')

    # Compute inverse mapping function
    colormap = np.linspace(0., 255., num=nbits+1, endpoint=True, dtype='float32')[::-1]
    def invert_mapping_func(x):
        return np.interp(x, colormap, np.arange(nbits))
        
    # Decode image iteratively
    slices = []
    for r in range(0, pad_height, step_shape[0]):
        for c in range(0, pad_width, step_shape[1]):

            # Slice current block and retrieve bitstream
            block = padded_img[r:r+window_shape[0], c:c+window_shape[1]]
            bits = block.flatten()
            
            # Separate compressed columns and decode them separately
            segment_cols = {}
            last_segment = None
            for ci in range(0, len(bits), nbits):
                
                # Extract compressed column
                col = bits[ci:ci+nbits]

                # Determine which segment is affected
                segid = ci // (window_shape[1] * nbits)
                if segid!= last_segment or not segment_cols:
                    last_segment = segid
                    start = segid * window_shape[1] * nbits
                    
                # Store column belonging to this segment
                try:
                    segment_cols[segid].append(col)
                except KeyError:
                    segment_cols[segid] = [col]
                
            # Reconstruct each segment
            reconstructed_segment = []
            for si, collist in sorted(segment_cols.items()):
                
                # Convert multiple columns to single byte array
                colbytes = ''.join([''.join(str(bit) for bit in col) for col in collist])
                colbits = np.array([int(b) for b in colbytes], dtype='int32')

                # Retrieve original column values
                colvals = invert_mapping_func(colbits)

                # Restore actual pixel locations
                rowinds = slice(start // (window_shape[1]),
                               start // (window_shape[1]) + window_shape[0])
                colinds = slice(si * window_shape[1], (si+1) * window_shape[1])
                bl = block[rowinds, colinds]
                bx = np.zeros((*bl.shape, 2))
                by = np.zeros_like(bx)
                bx[..., 0] = -2**nbits * colvals[..., 0] - bl[..., 0]
                bx[..., 1] = -2**nbits * colvals[..., 1] - bl[..., 1]
                by[..., 0] = -2**nbits * colvals[..., 2] - bl[..., 0]
                by[..., 1] = -2**nbits * colvals[..., 3] - bl[..., 1]
                normalizer = np.maximum(1, np.linalg.norm(bx, axis=-1)**2
                                       + np.linalg.norm(by, axis=-1)**2)
                bx /= normalizer.reshape((-1, 1))
                by /= normalizer.reshape((-1, 1))
                cx = np.array([[bx[i,:]*-bg for bg in np.arange(window_shape[0])]
                                for i in range(window_shape[0])])
                cy = np.array([[by[i,:]*-cg for cg in np.arange(window_shape[0])]
                                for i in range(window_shape[0])])
                dxs = cx + cx.T - np.diag(cx.diagonal())
                dys = cy + cy.T - np.diag(cy.diagonal())
                recons = np.vstack([np.hstack([dxs, dys]),
                                    np.hstack([-dys, dxs])])
                reconstructed_segment.append(recons.T)
                
            # Merge reconstructed segments together into final image
            recons_slice = np.hstack(reconstructed_segment)
            nfilled = count_nonzero(bits)
            expected_ncols = (width - c) // window_shape[1] * window_shape[1] // nbits
            if nfilled!= expected_ncols:
                raise ValueError("Block incomplete")
            slices.append(recons_slice)

    # Combine slices into final image
    assert all([(sl.shape==(padded_img.shape[0]-2*steps,
                             padded_img.shape[1]-2*steps,
                             2*(2*window_shape[0]+1)**2))
                 for sl in slices]), "Slice shapes do not match"
    recon_img = np.stack(slices, axis=-1).mean(axis=-1)
    return recon_img
    
    
    
if __name__ == '__main__':
    pass
```