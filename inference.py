import cv2
import numpy as np
from openvino.inference_engine import IECore
class TextDetector:
    
    def __init__(self) -> None:
        ie = IECore()
        self.network = ie.read_network(
            model="models/horizontal-text-detection-0001.xml",
            weights="models/horizontal-text-detection-0001.bin",
        )
        self.execution_net = ie.load_network(self.network, "CPU")
        
        self.input_layer = next(iter(self.execution_net.input_info))
        self.output_layer = next(iter(self.execution_net.outputs))
        
    def test(self, image_path, threshold=0.5):
        # reading image
        img = cv2.imread(image_path)
        
        # extracting model inputs: batch_size = 1, num_channels = 3 (RGB), height = 704, width = 704
        batch_size, num_channels, height, width = self.network.input_info[self.input_layer].tensor_desc.dims
        
        # resizing the input image to desired size
        resized_image = cv2.resize(img, (width, height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        
        # inferene on the input imahe
        result = self.execution_net.infer(inputs={self.input_layer: input_image})
        
        # output predictions
        predictions = result["boxes"]
        # removing no predictions  
        predictions_req = predictions[~np.all(predictions == 0, axis=1)]
        
        max_probability = 0
        
        # if the max prediciton probability exceeds the threshold then we return TEXT
        if len(predictions_req) != 0:
            for i in range(len(predictions_req)):
                max_probability = max(max_probability, predictions_req[i][4])
                
        if max_probability >= threshold:
            return "TEXT PRESENT"
            
        return "TEXT NOT PRESENT"
    
    
if __name__ == "__main__":
    obj = TextDetector()
    threshold = 0.5
    
    # Inferenceing image containing text
    text_image_path = "Music-To-Be-Murdered-By.jpg"
    print(obj.test(text_image_path, threshold))
    
    # Inferenceing image without text
    non_text_image_path = "knight-berserk-desktop-background.jpg"
    print(obj.test(non_text_image_path, threshold))
                
        
        