import cv2 as cv
import numpy as np

# This runs faster, because of the fact that does not finds connected aread for all the thresholds
# But, for each point you need to find a connected area to, you need to run the whole script again

# Dilates the masks

def show_image(image, name='', resize_min_height = 640):
	if image is None:
		print("The image is empty!")
		return
	image = image.copy()
	r = float(resize_min_height) / image.shape[1]
	dim = (resize_min_height, int(image.shape[0] * r))
	resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)
	cv.imshow(name, resized_image)
	cv.waitKey(0)
	cv.destroyAllWindows()

# image_path = '15.png'
image_path = 'cars.jpg'
# flare
x = 705
y = 570
# cloud
# x = 658
# y = 317

# блик
# x = 614
# y = 459
# солнце
# x = 460
# y = 260
# дебильных дебилов

image = cv.imread(image_path)
show_image(image)

image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
image_l = image_lab[:,:,0]
show_image(image_l)

l_threshold_radius = 0.1

def get_thresholds(pixel_luminosity, l_threshold_radius):
	converted_to_255_threshold = int(255 * l_threshold_radius)
	left = np.clip(pixel_luminosity - converted_to_255_threshold, 0, 255)
	right = np.clip(pixel_luminosity + converted_to_255_threshold, 0, 255)
	return left, right

def get_threshed_image(pixel_luminosity, image_l):
	thresholds = get_thresholds(pixel_luminosity, l_threshold_radius)
	threshed_from_bottom_and_top_image = np.array(
		((image_l <= thresholds[1]) * (image_l >= thresholds[0])) * 255, 
		dtype=np.uint8
		)
	return threshed_from_bottom_and_top_image

pixel_luminosity = image_l[y, x]
threshed_image = get_threshed_image(pixel_luminosity, image_l)
show_image(threshed_image)

def apply_circular_morph_operation(morphological_operation, image, kernel_size):
	dilatation_type = cv.MORPH_ELLIPSE
	kernel = cv.getStructuringElement(dilatation_type, (2*kernel_size + 1, 2*kernel_size+1), 
		(kernel_size, kernel_size))
	dilated_image = morphological_operation(image, kernel)
	return dilated_image

def apply_dilation(image, kernel_size):
	return apply_circular_morph_operation(cv.dilate, image, kernel_size)

threshed_image = apply_dilation(threshed_image, kernel_size=4)

def get_min_max_area_in_pixels(image, min_area_portion, max_area_portion):
	image_area = image.shape[0] * image.shape[1]
	min_area_in_pixels = min_area_portion * image_area
	max_area_in_pixels = max_area_portion * image_area
	return min_area_in_pixels, max_area_in_pixels

min_area, max_area = get_min_max_area_in_pixels(image, 1e-4, 1e-3)

def get_connected_components_list_from_image(image, min_area, max_area):
	filtered_connected_components = []
	num_labels, labels, statistics, centroids = get_raw_connected_components(image)
	for label in range(num_labels):
		label_area = statistics[label, cv.CC_STAT_AREA]
		if (label_area > min_area) and (label_area < max_area):
			label_mask = np.array((labels == label) * 255, dtype=np.uint8)
			filtered_connected_components.append({
				'mask' : label_mask,
				'area' : label_area,
				'centroids' : centroids[label]
				})
	return filtered_connected_components

def get_raw_connected_components(image):
	return cv.connectedComponentsWithStats(image, 4, cv.CV_32S)

connected_components = get_connected_components_list_from_image(threshed_image, min_area, max_area)

def get_pixel_theshold_window_mask(connected_components, x, y, pixel_luminosity):
	mask = None
	all_pixel_matching_connected_components = connected_components
	for connected_component in all_pixel_matching_connected_components:
		if connected_component['mask'][y, x]:
			mask = connected_component['mask']
	if mask is None:
		print('Warning: connected area withing given threshold does not exist for', "x", x, "y", y)
	return mask

mask = get_pixel_theshold_window_mask(connected_components, x, y, pixel_luminosity)
show_image(mask)