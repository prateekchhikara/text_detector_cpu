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
        
        self.colors = {"red": (0, 0, 255), "green": (0, 255, 0)}
        
    def draw_overlay(self, im_name, original_image, resized_image, predictions, res_folder = 'results/'):
        # Fetch image shapes to calculate ratio
        (real_y, real_x), (resized_y, resized_x) = original_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
        
        for box in predictions:
            # Pick confidence factor from last place in array
            conf = box[-1]
            
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            original_image = cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), self.colors["green"], 5)
            
            original_image = cv2.putText(original_image, f"{conf:.2f}", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["red"], 1, cv2.LINE_AA)
        
        cv2.imwrite(res_folder + im_name, original_image)
    
    def test(self, image_path, threshold=0.5):
        im_name = image_path.split('/')[-1]
        # reading image
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
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
        
        self.draw_overlay(im_name, img, resized_image, predictions_req)
        
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
    
    # Inferenceing image containing text (based on threshold)
    text_image_path = "ted_lasso.jpeg"
    print(obj.test(text_image_path, threshold))
    
    # Inferenceing image without text (based on threshold)
    non_text_image_path = "knight-berserk-desktop-background.jpg"
    print(obj.test(non_text_image_path, threshold))
                
        
        