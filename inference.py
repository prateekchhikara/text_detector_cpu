import cv2
from openvino.inference_engine import IECore


class TextDetector:
    
    def __init__(self) -> None:
        ie = IECore()
        self.network = ie.read_network(
            model = "",
            weights = ""
        )
        self.execution_net = ie.load_network(self.network, "CPU")
        
        self.input_layer = next(iter(self.execution_net.input_info))
        self.output_layer = next(iter(self.execution_net.outputs))
        
    def test(self, image_path, threshold=0.5):
        img = cv2.imread(image_path)
        batch_size, num_channels, height, width = net.input_info[input_layer_ir].tensor_desc.dims
        resized_image = cv2.resize(img, (width, height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        result = exec_net.infer(inputs={input_layer_ir: input_image})
        boxes = result["boxes"]
        boxes = boxes[~np.all(boxes == 0, axis=1)]
        
        max_probability = 0
        
        if len(boxes) != 0:
            for i in range(len(boxes)):
                max_probability = max(max_probability, boxes[i][4])
                
        if max_probability >= threshold:
            return "TEXT PRESENT"
            
        return "TEXT NOT PRESENT"
    
    
if __name__ == "__main__":
    obj = TextDetector()
    threshold = 0.5
    text_image_path = ""
    non_text_image_path = ""
    
    print(obj.test(text_image_path, threshold))
    print(obj.test(non_text_image_path, threshold))
                
        
        