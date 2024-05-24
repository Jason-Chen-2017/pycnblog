
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Metal 是 Apple 在 iOS、tvOS 和 macOS 操作系统上使用的底层图形 API。近年来，Metal 的性能越来越高，已经成为开发高性能游戏和 App 必备的工具。为了让 Metal 更加容易被开发者使用，Apple 提出了 MetalKit 扩展库，用于快速实现 Metal 编程。随着 iOS 15 和 MacOS Big Sur 发布，Vuforia 也发布了 MetalKIt 扩展库。Vuforia 使用 MetalKIt 提供了多种功能，如相机渲染、AR 技术支持等。本文将介绍 MetalKIt 扩展库，并介绍如何集成到 Xcode 项目中。另外，本文还会对 MetalKIt 中常用的功能组件进行详细介绍，并展示代码示例。希望通过阅读本文，Metal 开发者可以更好地理解 MetalKIt 扩展库及其功能特性，并结合自己的实际需求，去运用 Metal 创造出更加酷炫的 AR/VR 体验。
# 2.基本概念术语说明
## MetalKit 扩展库
MetalKit 是一个帮助 Metal 开发者更轻松地实现各种高级 Graphics 功能的框架。该框架提供了丰富的组件，包括照相机渲染、AR 技术支持、光线跟踪等。它具有模块化结构，可以单独使用某些组件，也可以按照不同的组合方式使用。MetalKit 扩展库支持以下功能：

1. 相机渲染：MetalKit 可以直接在 View 中渲染相机视野中的三维内容，同时支持相机视频流的处理，实现 AR 活体检测、环境光遮蔽等效果。

2. AR 技术支持：MetalKit 支持多种 AR 技术，例如特征点识别（QR Code、AR tags）、卡片扫描（NFC、iBeacon）等。还可以处理多个实时相机视图的混合渲染，提升 AR 体验的交互性和实时响应能力。

3. 基于物理的渲染：MetalKit 支持基于物理的渲染（PBR）技术，能够给物体赋予真实感的反射效果。

4. 模型渲染：MetalKit 支持加载并渲染多个 3D 模型文件，并支持自定义材质属性。

5. 渲染管道优化：MetalKit 提供渲染管道优化和自动管理机制，能够有效降低开发者的渲染负担。

6. VR 内容支持：MetalKit 为 iOS 和 iPadOS 提供 VR 内容的渲染支持，例如 Oculus Quest 和 HTC Vive。

7. 动画和特效：MetalKit 提供丰富的动画和特效组件，包括 Sprite Kit、POP、ReusePool 等。

## OpenGL ES vs Metal
OpenGL ES 是苹果公司于 2008 年推出的基于 OpenGL ES 规范的开源跨平台接口标准。Metal 是 Apple 在 iOS、tvOS 和 macOS 上使用的底层图形 API。两者之间最大的不同之处在于 Metal 是针对移动设备和桌面计算机设计的，而 OpenGL ES 是针对嵌入式系统设计的。在较短的历史时间里，OpenGL ES 在某些方面取得了成功，但随着 GPU 发展速度的提升，以及移动设备硬件的升级换代，iOS 9 和 macOS Mojave 将其抛弃了。因此，Metal 将成为 iOS、macOS 和 tvOS 下的首选图形 API。

除此之外，Metal 的性能要优于 OpenGL ES。Apple 在 WWDC 2017 上首次公开宣布 Metal 的性能超过 OpenGL ES 2.0。截至目前，Metal 的最新版本是 macOS Big Sur 的 Metal 2.0。Metal 和 OpenGL ES 在很多方面都有相同之处，比如可编程管线，数据布局等。但是，由于 Metal 有着比 OpenGL ES 更强大的性能表现，所以通常情况下使用 Metal 会比 OpenGL ES 更有效率。

## GLKit vs MetalKit
GLKit 是 Apple 在 iOS 之前推出的 OpenGL 图形库。它作为系统的一部分，支持常用的 2D、3D 渲染功能。从 iOS SDK 8 开始，Apple 开始全面转向 Metal ，而后者拥有更广泛的功能支持。因此，GLKit 不再受官方支持，而 MetalKit 则成为了最佳选择。

## Vuforia
Vuforia 是一款著名的 Augmented Reality (增强现实) 开发工具，它为开发者提供了完整的 AR 技术解决方案。Vuforia 在 iOS 设备上运行良好，主要原因是它使用了 Metal KIt 。Vuforia 通过 MetalKit 提供了 AR 技术支持，包括 6DOF（6 Degrees of Freedom，自由度）定位和图像识别。Vuforia 还支持光线追踪，具备高度灵活的定制能力。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
MetalKit 扩展库主要由如下几个主要的组件组成：

- CameraRenderer: 相机渲染组件，用于渲染相机视野中的三维内容。

- Renderer: 渲染组件，提供渲染功能支持，包括绘制图片和三维对象。

- ARView: AR视图组件，提供基于 Metal 的 AR 技术支持，包括世界坐标系、摄像头坐标系、平面坐标系、平行光模型等。

- DetectionManager: 检测管理器组件，提供基于特征点的图像识别和环境光遮蔽检测功能。

- PhysicsWorld: 物理世界组件，提供物理渲染支持，包括碰撞检测、模拟、模糊模拟等。

- VideoBackgroundConfigration: 视频背景配置组件，提供播放本地或网络视频文件的支持。

每个组件都有相应的功能和方法，本节将详细介绍这些组件。

## CameraRenderer
CameraRenderer 是 MetalKit 扩展库中最基础的组件。它的作用是渲染相机视野中的三维内容。CameraRenderer 有两个重要的方法：

1. renderCamera: 这个方法用于渲染相机视野中的三维内容。

2. processVideoFrame: 这个方法用于处理相机视频流。

### renderCamera
renderCamera 方法使用指定的渲染模式渲染相机视野中的三维内容。渲染模式可以是沿着视角方向渲染（正投影模式）或者透视投影渲染（透视投影模式）。相机的视角可以通过设置 projectionMatrix 属性来设置。

CameraRenderer 还可以利用 Light 类来创建光源，包括方向光、点光源、聚光灯等。Light 类可以设置位置和颜色信息，并指定光照强度。

CameraRenderer 还可以使用 Material 类来创建材质，包括金属、非金属材料、镜面材料等。Material 类可以设置纹理贴图、颜色、透明度、反射度等属性。

### processVideoFrame
processVideoFrame 方法用于处理相机视频流。它接受一个 CMSampleBufferRef 对象作为参数，这个对象表示相机视频流。CMSampleBufferRef 表示编码的媒体样本缓冲区，包含媒体数据的原始格式、大小和时间戳信息。调用 processVideoFrame 方法可以实现视频流的处理，包括帧速率转换、视频的缩放、视频的叠加、视频的旋转等。

## Renderer
Renderer 是 MetalKit 扩展库中第二个基础组件。它的作用是在屏幕上渲染图片、三维物体。Renderer 有两个重要的方法：

1. drawAtTime: 这个方法用于渲染屏幕上的图片、三维物体。

2. setRenderPipelineState: 这个方法用于设置渲染管线状态。

### drawAtTime
drawAtTime 方法用于渲染屏幕上的图片、三维物体。调用 drawAtTime 方法可以传入当前时间，Renderer 根据当前时间确定当前需要渲染哪些对象。

### setRenderPipelineState
setRenderPipelineState 方法用于设置渲染管线状态。调用 setRenderPipelineState 方法可以改变渲染管线的状态，如改变光照的颜色、饱和度、亮度等。

## ARView
ARView 是 MetalKit 扩展库中第三个基础组件。它的作用是提供基于 Metal 的 AR 技术支持，包括世界坐标系、摄像头坐标系、平面坐标系、平行光模型等。ARView 有两个重要的方法：

1. updateAtTime: 这个方法用于更新 AR 视图状态。

2. projectPointsFromModelToScreenSpace: 这个方法用于把模型空间下的点投影到屏幕空间中。

### updateAtTime
updateAtTime 方法用于更新 AR 视图状态。调用 updateAtTime 方法可以传入当前时间，ARView 根据当前时间更新相机的位置、相机的朝向等信息。

### projectPointsFromModelToScreenSpace
projectPointsFromModelToScreenSpace 方法用于把模型空间下的点投影到屏幕空间中。调用 projectPointsFromModelToScreenSpace 方法可以把模型空间下的点投影到屏幕空间中，返回屏幕坐标系下的点。

## DetectionManager
DetectionManager 是 MetalKit 扩展库中第四个基础组件。它的作用是提供基于特征点的图像识别和环境光遮蔽检测功能。DetectionManager 有三个重要的方法：

1. detect: 这个方法用于识别图像特征点。

2. findPlane: 这个方法用于查找平面信息。

3. estimateLightingIntensityInImageWithCubeMap: 这个方法用于估计环境光遮蔽效果。

### detect
detect 方法用于识别图像特征点。调用 detect 方法可以识别图像中所有的特征点，并根据特征点的类型和位置设置相应的标记。

### findPlane
findPlane 方法用于查找平面信息。调用 findPlane 方法可以查找图像中的平面信息，并判断平面的位置和法向量。

### estimateLightingIntensityInImageWithCubeMap
estimateLightingIntensityInImageWithCubeMap 方法用于估计环境光遮蔽效果。调用 estimateLightingIntensityInImageWithCubeMap 方法可以估计图像的环境光遮蔽效果，并返回结果值。

## PhysicsWorld
PhysicsWorld 是 MetalKit 扩展库中第五个基础组件。它的作用是提供物理渲染支持，包括碰撞检测、模拟、模糊模拟等。PhysicsWorld 有两个重要的方法：

1. simulateWithDeltaTime: 这个方法用于物理模拟。

2. blurDepthAndNormal: 这个方法用于模糊渲染。

### simulateWithDeltaTime
simulateWithDeltaTime 方法用于物理模拟。调用 simulateWithDeltaTime 方法可以让物体做相应的运动变化。

### blurDepthAndNormal
blurDepthAndNormal 方法用于模糊渲染。调用 blurDepthAndNormal 方法可以让物体进行模糊渲染，让渲染效果看起来更平滑。

## VideoBackgroundConfiguration
VideoBackgroundConfiguration 是 MetalKit 扩展库中第六个基础组件。它的作用是提供播放本地或网络视频文件的支持。VideoBackgroundConfiguration 有两个重要的方法：

1. startPlayingVideo: 这个方法用于播放本地或网络视频文件。

2. stopPlayingVideo: 这个方法用于停止播放视频文件。

### startPlayingVideo
startPlayingVideo 方法用于播放本地或网络视频文件。调用 startPlayingVideo 方法可以播放指定的视频文件，并显示在屏幕上。

### stopPlayingVideo
stopPlayingVideo 方法用于停止播放视频文件。调用 stopPlayingVideo 方法可以停止正在播放的视频文件。

# 4.具体代码实例和解释说明
## 创建相机视图
相机视图组件 CameraRenderer 可以渲染相机视野中的三维内容。下面我们创建一个相机视图组件，并添加到 UIView 中。

```swift
class ViewController: UIViewController {

    @IBOutlet weak var cameraView: UIView!
    
    private let renderer = Renderer()
    private let cameraRenderer = CameraRenderer(view: self.cameraView)
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 设置渲染器
        DispatchQueue.main.async {
            self.renderer.delegate = self
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.colorAttachments[0].pixelFormat =.bgra8Unorm

            if #available(iOS 12.0, *) {
                let library = MTLDevice.default().newDefaultLibrary!
                let vertexFunctionName = "vertexShader"
                let fragmentFunctionName = "fragmentShader"

                guard let vertexFunction = try? library.newFunctionWithName(vertexFunctionName),
                    let fragmentFunction = try? library.newFunctionWithName(fragmentFunctionName) else {
                        fatalError("Failed to get vertex or fragment function from library")
                }
                
                pipelineDescriptor.vertexFunction = vertexFunction
                pipelineDescriptor.fragmentFunction = fragmentFunction
            }

            do {
                self.renderer.renderPipelineState = try renderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            } catch {
                fatalError("\(error)")
            }
        }
    }
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()

        let size = UIScreen.main.bounds.size
        let aspectRatio = min(size.width / size.height, 1.0)
        
        self.cameraRenderer.projectionMatrix = matrix_perspectiveProjectionForBounds((0.0, -0.5, 0.0),
                                                                                  (aspectRatio, 0.5, 1.0),
                                                                                  45.0 * NSIMD_PI_F / 180.0,
                                                                                  0.01, 1000.0);
                
        self.cameraRenderer.isRenderingPaused = false
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        NotificationCenter.default.addObserver(self,
                                               selector:#selector(ViewController.videoOutputReadyHandler(notification:)),
                                               name:.AVFoundationVideoDecoderNewPixelBufferNotification,
                                               object:nil)
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}
```

上面代码中，我们首先声明了一个 `Renderer` 对象，用来渲染屏幕上的图片和三维物体。然后，我们创建了一个 `CameraRenderer` 对象，设置了渲染目标，并将其加入到了 UIView 中。最后，我们设置了相机的视角矩阵，使得相机可以看到完整的场景。我们还订阅了通知 `.AVFoundationVideoDecoderNewPixelBufferNotification`，当新的像素数据可用时，通知中心会发送该通知。

```swift
func videoOutputReadyHandler(notification: NSNotification) {
    let sampleBuffer = notification.object as? CMSampleBuffer

    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer!) else { return }

    switch CMGetAttachment(pixelBuffer!, kCGImagePropertyOrientation, nil) {
    case.up: break
    default:
        let transform = CGAffineTransform(rotationAngle: CGFloat(-Double.pi))
        CFRelease(CMCopyDescription(transform))
        CMRotateImage(pixelBuffer, transform, nil)
        break
    }

    DispatchQueue.main.async {
        self.cameraRenderer.videoTexture = AVCaptureVideoPreviewLayer(session: AVCaptureSession())?.texture(with: pixelBuffer)!
    }
}
```

上面的代码处理视频流，并将视频数据显示在相机视图中。该代码采用了 AVCaptureVideoPreviewLayer 来显示视频，并根据视频数据调整渲染模式，确保视频正确显示。

## 添加对象到相机视图
渲染组件 Renderer 提供了渲染功能支持，包括绘制图片和三维对象。我们可以在 Renderer 对象上绘制一个矩形框，来验证是否正常工作。

```swift
let cubeSize = CGSize(width: 2.0, height: 2.0)
var position = Vector3(x: 0.0, y: -cubeSize.height / 2.0, z: -cubeSize.width / 2.0)
position = cameraRenderer.convertPositionFromModelToCameraCoordinateSystem(position)

renderer.addCubeWithSize(cubeSize, position: position)
```

上面的代码添加了一个 2 x 2 尺寸的立方体对象到相机视图中，并设置它的位置，使得它位于相机前方。

## 图像识别
检测管理器组件 DetectionManager 提供基于特征点的图像识别和环境光遮蔽检测功能。我们可以调用 detect 方法来识别图像中所有的特征点，并根据特征点的类型和位置设置相应的标记。

```swift
private let detectionManager = DetectionManager()

override func viewDidLoad() {
   ...
    
    DispatchQueue.global(qos:.background).async { [weak self] in
        let image = self?.detectionManager.captureStillImage()
        DispatchQueue.main.async {
            self?.displayDetectedElements(image: image)
        }
    }
    
   ...
}

extension ViewController: DetectionManagerDelegate {
    func displayDetectedElements(image: UIImage?) {
        guard let image = image else { return }
        
        detectionManager.setImage(image)
        detectionManager.detect()
    }
}
```

上面的代码订阅了 `NSNotification.name(.AVFoundationVideoDecoderNewPixelBufferNotification)`，并在收到通知时获取当前视频帧。然后，我们调用 `detectionManager` 的 `captureStillImage` 方法获取当前相机帧的图片，并将图片传给 `detectionManager` 的 `setImage` 方法。接下来，我们调用 `detectionManager` 的 `detect` 方法识别特征点。

```swift
func detect() {
    DispatchQueue.global(qos:.userInitiated).async { [weak self] in
        guard let image = self?.detectionManager.image,
            let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue() else { return }

        do {
            self?.detectionManager.detectedElements.removeAll()

            let bufferAllocator = MTKCommandBufferAllocator.init(device: device)
            let commandBuffer = bufferAllocator.commandBuffer(priority:.default)
                
            self?.detectionManager.commandEncoder = commandBuffer!.computeCommandEncoder

            self?.renderEnvironmentMap(commandBuffer: commandBuffer)

            if!self?.detectionManager.disablePlaneFinding?? true {
                self?.findPlanes()
            }

            if!self?.detectionManager.disableObjectRecognition?? true {
                self?.recognizeObjects()
            }

            if!self?.detectionManager.disableAmbientOcclusionEstimation?? true {
                self?.estimateLightingIntensity(in: image, using: commandBuffer)
            }

            try commandBuffer.endEncoding()
            commandBuffer.commit()

            DispatchQueue.main.async { [weak self] in
                self?.updateDetectionResults(in: image)
            }
        } catch {
            print("An error occurred during the recognition process:\n\(error)")
        } finally {
            bufferAllocator.release()
        }
    }
}

func renderEnvironmentMap(commandBuffer: MTLCommandBuffer) {
    guard let environmentMap = EnvironmentMap.sharedInstance.environmentMap,
        let kernelBundle = Bundle(for: type(of: self)),
        let resourcePath = bundleResourcePath(bundleName: kernelBundle.bundleIdentifier! + ".metallib"),
        let data = Data(contentsOf: URL(fileURLWithPath: resourcePath), options:.mappedRead)?.bytes else {
            fatalError("Failed to load environment map.")
    }
            
    let program = device.newProgram(data: data)
    let uniform = program.uniform(at: "environmentMapSampler")
            
    for textureSetIndex in 0..<kNumCubeMapMipmapLevels {
        guard let levelSize = Int(pow(2.0, Double(textureSetIndex))) else { continue }
            
        for faceID in 0..<6 {
            let textureSet = environmentMap.textureSets[(faceID * kNumCubeMapMipmapLevels) + textureSetIndex]
            let texture = textureSet.texture
            
            commandBuffer.parallelRenderCommandEncoder(withDescriptor: MTLRenderCommandEncoderDescriptor(),
                                                           perform: { (encoder) -> Void in
                                encoder.setFragmentTexture(texture, at: 0)
                                encoder.setFragmentSamplerState(texture.samplerState, at: 0)

                                encoder.setCullMode(.back)
                                encoder.setFrontFacingWinding(.counterClockwise)
                                encoder.setDepthStencilState(depthStencilState, writeMask: 0xFFFFFFFF)
                                
                                encoder.setVertexBuffer(vertexBuffer, offset: 0, at: 0)
                                encoder.drawPrimitives(.triangleStrip, vertexStart: 0, vertexCount: 4)
                            })
        }
    }
}

private func recognizeObjects() {
    guard let context = CIContext(),
        let detector = try?VNRecognizeTextRequest.engine(),
        let image = detectionManager.image else { return }
        
    let scaleFactor = UIScreen.main.scale
    
    let visionImage = CIImage(cvImage: CVImageBufferGetOpenGLESCompatibilityTexture(CVOpenGLTextureCacheGetTypeID()),
                              options: [:])
    let orientation = CGImagePropertyOrientation.topLeft.rawValue |
                      CGImagePropertyOrientation.bottomRight.rawValue |
                      CGImagePropertyOrientation.leftTop.rawValue |
                      CGImagePropertyOrientation.rightBottom.rawValue
                      
    let visionOptions = [kCIInputOrientationKey : NSNumber(value:orientation)]
    
    detector?.enqueue(visionImage, options: visionOptions, completionHandler: {[weak self](request, results, error) in
        guard let resultsArray = results as? [VNRecognizedText],
            let result = resultsArray.first else {
                debugPrint("[VisionKit] No text recognized.")
                return
        }
        
        let bounds = CGRectMake(result.boundingBox.origin.x * scaleFactor,
                               -(result.boundingBox.origin.y + result.boundingBox.size.height) * scaleFactor,
                               result.boundingBox.size.width * scaleFactor,
                               result.boundingBox.size.height * scaleFactor)
                            
        guard let layer = detectionManager.detectedElementsLayers?.last else {
            let layer = CALayer()
            layer.frame = bounds
            layer.backgroundColor = UIColor.white
            detectionManager.detectedElementsLayers?.append(layer)
            detectionManager.view?.layer.insertSublayer(layer, below: detectionManager.view?.layer.sublayers?[0])
            return
        }
            
        layer.frame = bounds
        layer.borderWidth = 1.0
        layer.borderColor = UIColor.lightGray.cgColor
        
        CATransaction.begin()
        CATransaction.setValue(CATransform3DIdentity, forKey: kCATransform3DForKey)
        CATextLayer.appearance().setText(result.topCandidates![0].string)
        CATextLayer.appearance().font = UIFont.systemFont(ofSize: 20.0)
        CATextLayer.appearance().textColor = UIColor.black.cgColor
        CATextLayer.appearance().textAlignment =.center
        CATextLayer.appearance().shadowOffset = CGSize(width: 1.0, height: 1.0)
        CATextLayer.appearance().shadowRadius = 1.0
        CATextLayer.appearance().shadowColor = UIColor.gray.cgColor
        CATextLayer.appearance().shadowOpacity = 1.0
        CATextLayer.appearance().enablesAntialiasing = true
        CATextLayer.appearance().allowsFontScaling = true
        CATransaction.commit()
    })
}
```

上面的代码调用 `captureStillImage` 方法获取当前相机帧的图片，并将图片传给 `detectionManager` 的 `setImage` 方法。然后，我们调用 `detectionManager` 的 `detect` 方法识别特征点。在识别完成后，我们画出识别结果的边界框，并显示在 UIView 上。

## 平面查找
检测管理器组件 DetectionManager 提供查找平面信息的功能。我们可以调用 `findPlane` 方法查找图像中的平面信息，并判断平面的位置和法向量。

```swift
if!disablePlaneFinding && planeDetectionAllowed {
    findPlanes()
}
    
func findPlanes() {
    DispatchQueue.global(qos:.userInitiated).async { [weak self] in
        guard let image = self?.detectionManager.image,
            let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue() else { return }

        do {
            self?.detectionManager.planes.removeAll()

            let bufferAllocator = MTKCommandBufferAllocator.init(device: device)
            let commandBuffer = bufferAllocator.commandBuffer(priority:.default)
                
            self?.detectionManager.commandEncoder = commandBuffer!.computeCommandEncoder

            if let featurePointDetector = self?.detectionManager.featurePointDetector,
                let threshold = featurePointDetector.maximumFeaturePointsPerFrame {
                
                // Check if there are enough points for a planar model
                let numPoints = featurePointDetector.numVisibleFeaturePoints
                if numPoints < threshold {
                    DispatchQueue.main.async {
                        let message = NSString(format: NSLocalizedString("Not enough visible features found (%d/%d).", comment: ""),
                                                 numPoints, threshold)
                        
                        self?.presentAlert(title: NSLocalizedString("Insufficient Features Found", comment: ""),
                                           message: String(message),
                                           cancelButtonTitle: "",
                                           otherButtonTitles: [],
                                           tapHandler: nil)
                    }

                    // Release resources used by the feature point detector and exit early
                    self?.detectionManager.featurePointDetector = nil
                    
                    return
                }
            }

            // Render the scene with an orthographic projection for finding planes only
            renderScene(commandBuffer: commandBuffer, isOrthographicProjection: true)
            
            // Find planes by extracting features from rendered depth values
            if let orthographicDepthExtractor = self?.detectionManager.orthographicDepthExtractor {
                orthographicDepthExtractor.extractFeatures(commandBuffer: commandBuffer)
            }
            
            // Transform extracted features into their corresponding planes
            if let currentImage = self?.detectionManager.currentImage {
                extractPlanes(fromImage: currentImage)
            }

            try commandBuffer.endEncoding()
            commandBuffer.commit()

            DispatchQueue.main.async { [weak self] in
                self?.postProcessPlanes()
            }
        } catch {
            print("An error occurred while finding planes:\n\(error)")
        } finally {
            bufferAllocator.release()
        }
    }
}
```

上面的代码查找图像中的平面信息。首先，我们检查当前帧是否有足够的特征点用于建立平面模型，如果没有足够的特征点，我们就不执行平面查找，并提示用户重新拍摄。

然后，我们使用渲染器渲染场景，但使用正交投影的方式来仅查找平面信息。接下来，我们调用 `orthographicDepthExtractor` 的 `extractFeatures` 方法来查找深度信息，并将其转换为平面信息。接着，我们使用 `extractPlanes` 方法从当前帧中提取平面信息。

最后，我们更新渲染器渲染环境信息。渲染完毕之后，我们调用 `postProcessPlanes` 方法对渲染好的平面信息进行后处理。

```swift
func postProcessPlanes() {
    var normalsToTest: [(Vector3, Float)] = []
    for (plane, _) in planes where canUseForPrediction {
        let distance = (plane.normal * (-cameraPosition)).magnitude
        if abs(distance) > minimumDistanceThresholdForPrediction {
            normalsToTest.append((plane.normal, distance))
        }
    }
    
    DispatchQueue.global(qos:.userInitiated).async { [weak self] in
        var predictions: [Any] = []
        for (_, distance) in normalsToTest {
            let predictionResult = try? VNCoreMLModel.prediction(for: inputImage,
                                                                  using: self?.coremlClassifier,
                                                                  labels: ["plane"],
                                                                  confidenceThreshold: 0.5,
                                                                  completionHandler: {
                                                                        DispatchQueue.main.async {
                                                                            if let predictionResult = $0 {
                                                                                predictions.append(predictionResult)
                                                                            }
                                                                        }
                                                                    })
        }

        DispatchQueue.main.async { [weak self] in
            self?.handlePredictions(predictions: predictions)
        }
    }
}
```

上面的代码使用 Core ML 分类器预测出当前帧中的平面信息。我们遍历找到的所有平面，如果距离相机远离，且平面的法向量满足一定条件，我们就认为这是一个可能的平面候选。

对于每个可能的平面候选，我们都会使用 Core ML 模型进行预测，并得到一个置信度分数。如果置信度超过某个阈值，我们就认为当前帧中存在这个平面。否则，我们跳过这个候选。

```swift
func handlePredictions(predictions: [Any]) {
    var newPlanes: [VNPlane] = []
    for prediction in predictions {
        guard let dictionary = prediction as? [String: Any] else { continue }
        
        guard let center = dictionary["center"] as? [Double],
            let width = dictionary["width"] as? Double,
            let length = dictionary["length"] as? Double,
            let normal = dictionary["normal"] as? [Double] else { continue }
            
         let origin = Vector3(x: center[0], y: center[1], z: center[2])
         let extent = Extent(width: width, length: length)
         let rotation = Quaternion(vector: Vector3(x: normal[0], y: normal[1], z: normal[2]), angle: 0.0)
         
         let plane = VNPlane(center: origin, extent: extent, rotation: rotation,
                          topLeftCorner: Vector3(),
                          bottomRightCorner: Vector3(),
                          leftUpRightDownVectors: nil)

         newPlanes.append(plane)
    }
        
    self.planes = newPlanes
    DispatchQueue.main.async {
        // Update rendering with new planes
        DispatchQueue.main.async {
            self?.renderer.scene = Scene(objects: objects, lights: lights, planes: newPlanes)
        }
    }
}
```

上面的代码处理 Core ML 预测结果，并生成新的平面实例。我们遍历所有 Core ML 模型预测出的结果，如果置信度超过某个阈值，我们就认为这是一条有效的平面候选。接着，我们构造 `VNPlane` 对象，并将其添加到 `planes` 数组中。

最后，我们更新渲染器渲染状态，并渲染出带有新平面信息的新场景。