import argparse
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from typing import Dict, Any, List, Union
from tqdm import tqdm
import shutil
import tempfile

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def _set_device(args):
    if args.device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def sam2_instance(args):

    sam2 = build_sam2(args.model_cfg, args.sam2_checkpoint, device=args.device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model = sam2,
        points_per_batch=6,
        pred_iou_thresh=0.9,
        # stability_score_thresh=0.97,
    )

    return mask_generator

def data_instance(args):
    # '/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json'
    data = json.load(open(args.data_path))

    return data

class SegmentationDataManager:
    def __init__(self, base_directory: str):
        """
        Initialize the manager with a base directory for storing all data.
        
        Args:
            base_directory (str): Directory where all data will be stored
        """
        self.base_directory = base_directory
        self.arrays_dir = os.path.join(base_directory, "arrays")
        self.metadata_file = os.path.join(base_directory, "metadata.json")
        self.backup_file = os.path.join(base_directory, "metadata_backup.json")
        
        # Create directories if they don't exist
        os.makedirs(self.arrays_dir, exist_ok=True)
        
        # Initialize or recover metadata file
        self._initialize_or_recover_metadata()

    def _safe_json_read(self, filepath: str) -> Dict:
        """
        Safely read JSON file with error handling and recovery.
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Try to recover the JSON by reading until the last valid line
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Find the last complete object (ending with })
            last_brace_index = content.rfind('}')
            if last_brace_index != -1:
                valid_content = content[:last_brace_index + 1]
                try:
                    return json.loads(valid_content)
                except json.JSONDecodeError:
                    pass
            
            # If recovery failed, return empty dict
            return {}

    def _safe_json_write(self, data: Dict, filepath: str) -> None:
        """
        Safely write JSON file using a temporary file.
        """
        # Create a temporary file in the same directory
        temp_dir = os.path.dirname(filepath)
        with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as tf:
            # Write to temporary file
            json.dump(data, tf, indent=2)
            temp_filepath = tf.name
        
        # Rename temporary file to target file (atomic operation)
        shutil.move(temp_filepath, filepath)

    def _initialize_or_recover_metadata(self) -> None:
        """
        Initialize metadata file or recover from backup if main file is corrupted.
        """
        metadata = {}
        
        # Try to read main metadata file
        if os.path.exists(self.metadata_file):
            metadata = self._safe_json_read(self.metadata_file)
        
        # If main file is empty or corrupted, try backup
        if not metadata and os.path.exists(self.backup_file):
            metadata = self._safe_json_read(self.backup_file)
        
        # Save recovered or empty metadata
        self._safe_json_write(metadata, self.metadata_file)
        # Create backup
        self._safe_json_write(metadata, self.backup_file)

    def list_image_keys(self):
        """
        Check if an image has already been processed and saved.
        """
        metadata = self._safe_json_read(self.metadata_file)
        image_keys_list = []  
        for image_key, val in metadata.items():
            image_keys_list.append(image_key)

        return image_keys_list
    
    
    def save_data(self, image_key: str, segmentations: List[Dict[str, Any]]) -> None:
        """
        Save multiple segmentation data for a single image.
        """
        # Create image directory
        image_arrays_dir = os.path.join(self.arrays_dir, image_key)
        os.makedirs(image_arrays_dir, exist_ok=True)
        
        # Load existing metadata
        metadata = self._safe_json_read(self.metadata_file)
        
        # Initialize or get existing image metadata
        metadata[image_key] = []
        
        # Save each segmentation
        for idx, seg_dict in enumerate(segmentations):
            # Save the numpy array
            array_filename = f"segmentation_{idx}.npy"
            array_path = os.path.join(image_arrays_dir, array_filename)
            np.save(array_path, seg_dict['segmentation'])
            
            # Prepare metadata
            meta_entry = seg_dict.copy()
            meta_entry['segmentation'] = array_filename
            
            # Add metadata for this segmentation
            metadata[image_key].append(meta_entry)
        
        # Save updated metadata and backup
        self._safe_json_write(metadata, self.metadata_file)
        self._safe_json_write(metadata, self.backup_file)
        
    def process_dataset(self, data: List[Dict[str, str]], mask_generator: Any) -> None:
        """
        Process a dataset of images, generating and saving masks only for unprocessed images.
        """
        image_keys = self.list_image_keys()

        for i in tqdm(range(len(data))):
            # Get image key
            image_key = data[i]['image'].strip('/mnt/nushare2/data/baliao/multimodal/data/')
            
            # Check if image has already been processed
            if image_key in image_keys:
                continue
            
            try:
                # Get image path
                image_path = data[i]['image']
                    
                # Process image and generate masks
                image = Image.open(image_path)
                image = np.array(image.convert("RGB"))
                masks = mask_generator.generate(image)
                
                # Save masks
                self.save_data(image_key, masks)
                
            except Exception as e:
                print(f"Error processing image {image_key}: {str(e)}")
                continue

    
    def load_data(self, image_key: str, indices: Union[int, List[int]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load segmentation data for a specific image.
        """
        metadata = self._safe_json_read(self.metadata_file)
            
        if image_key not in metadata:
            raise KeyError(f"No data found for image: {image_key}")
        
        image_metadata = metadata[image_key]
        image_arrays_dir = os.path.join(self.arrays_dir, image_key)
        
        # Handle single index
        if isinstance(indices, int):
            metadata = image_metadata[indices].copy()
            array_path = os.path.join(image_arrays_dir, metadata['segmentation'])
            metadata['segmentation'] = np.load(array_path)
            return metadata
        
        # Handle multiple indices or None (all segmentations)
        load_indices = indices if indices is not None else range(len(image_metadata))
        result = []
        
        for idx in load_indices:
            metadata = image_metadata[idx].copy()
            array_path = os.path.join(image_arrays_dir, metadata['segmentation'])
            metadata['segmentation'] = np.load(array_path)
            result.append(metadata)
        
        return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", type=str, default= "/data/chatgpt/notebooks/mnulli/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="/data/chatgpt/notebooks/mnulli/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--base_directory", type=str, default='/data/chatgpt/notebooks/mnulli/sam2/notebooks/segmentation_data')
    parser.add_argument("--data_path", type=str, default='/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json')


    args = parser.parse_args()

    manager = SegmentationDataManager(args.base_directory)

    mask_generator = sam2_instance(args)

    data = data_instance(args)

    manager.process_dataset(data, mask_generator)

