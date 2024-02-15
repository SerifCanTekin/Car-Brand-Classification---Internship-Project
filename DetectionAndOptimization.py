import numpy as np
import csv
import argparse
import os
import tensorflow as tf
from PIL import Image
from io import BytesIO
import shutil
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

def replace_semicolon(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    updated_content = content.replace(';;;', ';')
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(updated_content)

def replace_semicolon_sec(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(directory, filename)
            replace_semicolon(file_path)
            print(f"Processed: {file_path}")

def get_csv_filenames(directory):
    csv_filenames = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".csv"):
            csv_filenames.append(os.path.join(directory, filename))
    return csv_filenames

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def read_csv_and_get_alfa(csv_files):
    alfa = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as csv_file_t:
            csv_reader = csv.reader(csv_file_t, delimiter=';')
            for fields in csv_reader:
                alfa.append(fields)
    return alfa


def temper(beta):
    gama = []
    for ele in beta:
        temp = ele[0:5]
        gama.append(temp)
    return gama

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: a file path (this can be local or on colossus)

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(model, image):
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)


    # Filter out only vehicle detections
    car_indices = np.where((output_dict['detection_classes'] == 3) | (output_dict['detection_classes'] == 4) | (output_dict['detection_classes'] == 6) | (output_dict['detection_classes'] == 8))[0]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][car_indices]
    output_dict['detection_classes'] = output_dict['detection_classes'][car_indices]
    output_dict['detection_scores'] = output_dict['detection_scores'][car_indices]

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, first5):
        for temp in first5:
            i_path = search_file('inputs/data', temp[4])
            if i_path != None:
                image_np = load_image_into_numpy_array(i_path)
                # Actual detection.
                output_dict = run_inference_for_single_image(model, image_np)

                i_path_name = os.path.basename(i_path)


                # Filter detections based on min_score_threshold
                min_score_threshold = 0.5
                above_threshold_indices = np.where(output_dict['detection_scores'] >= min_score_threshold)[0]
                output_dict['detection_boxes'] = output_dict['detection_boxes'][above_threshold_indices]
                output_dict['detection_classes'] = output_dict['detection_classes'][above_threshold_indices]
                output_dict['detection_scores'] = output_dict['detection_scores'][above_threshold_indices]

                if len(output_dict['detection_boxes']) > 0:
                    volumes = [get_box_volume(box) for box in output_dict['detection_boxes']]
                    largest_box_index = np.argmax(volumes)
                    largest_box = output_dict['detection_boxes'][largest_box_index]

                    # Keep only the largest detection box
                    output_dict['detection_boxes'] = np.array([largest_box])
                    output_dict['detection_classes'] = np.array([output_dict['detection_classes'][largest_box_index]])
                    output_dict['detection_scores'] = np.array([output_dict['detection_scores'][largest_box_index]])

                for box in output_dict['detection_boxes']:
                    ymin, xmin, ymax, xmax = box
                    im_height, im_width, _ = image_np.shape

                    # Koordinatları piksel cinsinden dönüştürün
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                  ymin * im_height, ymax * im_height)

                    # Kesilmiş görüntüyü alın
                    cropped_image = image_np[int(top):int(bottom), int(left):int(right)]

                    # Kesilmiş görüntüyü yeni bir dosyaya kaydedin

                    new_image_path = "outputs/cropped/{}".format(i_path_name)
                    os.makedirs("outputs/cropped", exist_ok=True)
                    Image.fromarray(cropped_image).save(new_image_path)



            else: continue




def get_box_volume(box):
    """Calculates the volume of the 3D box defined by the corners.

    Args:
        box (numpy array): Box corners in the format [ymin, xmin, ymax, xmax].

    Returns:
        float: The volume of the box.
    """
    ymin, xmin, ymax, xmax = box
    height = ymax - ymin
    width = xmax - xmin
    depth = 1.0  # Assuming 1 unit depth for 2D boxes
    volume = height * width * depth
    return volume




def converter(alfa):
    for ele in alfa:
        ele[4] = os.path.basename(ele[4])
    return alfa




def search_file(start_directory, target_file):
    for root, dirs, files in os.walk(start_directory):
        if target_file in files:
            return os.path.join(root, target_file)
    return None




def transporter(file_list):
    for element in file_list:
        source_path = search_file('outputs\\cropped', element[1])
        target_dir = os.path.join('outputs\\brands', element[0])
        target_path = os.path.join(target_dir, element[1])

        os.makedirs(target_dir, exist_ok=True)  # Create target directory if it doesn't exist

        try:
            shutil.copy(source_path, target_path)  # Copy instead of move
        except Exception as e:
            continue





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in images using Object Detection API')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model directory')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to the label map')
    args = parser.parse_args()

    target_directory_for_cvs = "inputs\\csv_files"  # Dizin yolunu burada belirtin
    replace_semicolon_sec(target_directory_for_cvs)

    temper_path = get_csv_filenames("inputs\\csv_files")
    all_items = read_csv_and_get_alfa(temper_path)
    first5 = temper(all_items)
    first5_wanted = converter(first5)
    detection_model = load_model(args.model)
    category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    index_of_target = first5_wanted.index(['12557', 'Infiniti', 'QX4', '0', '17512552.jpg'])
    del first5_wanted[:(index_of_target + 2)]
    run_inference(detection_model, category_index, first5_wanted)
    wanted = converter(all_items)
    result = [[sublist[1], sublist[4]] for sublist in wanted]
    transporter(result)
