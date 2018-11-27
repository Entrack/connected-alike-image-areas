import cv2 as cv
import numpy as np

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


class ConnectedAreaThresholdSelector():
	def __init__(
		self, image_path, 
		l_threshold_radius = 0.05, 
		min_area_portion = 0.0001, 
		max_area_portion = 0.1,
		kernel_size_erosion = 1,
		kernel_size_dilation = 4
		):
		self.l_threshold_radius = l_threshold_radius
		self.min_area = None
		self.max_area = None
		self.kernel_size_erosion = kernel_size_erosion
		self.kernel_size_dilation = kernel_size_dilation

		self.image = cv.imread(image_path)
		self.image_l = cv.cvtColor(self.image, cv.COLOR_BGR2LAB)[:,:,0]

		self.min_area, self.max_area = self.get_min_max_area_in_pixels(
			min_area_portion = min_area_portion, 
			max_area_portion = max_area_portion)
		print(self.__class__.__name__, "inited")

		print("Calculating all connected components for all the thresholds...")
		"""
		CONNECTED_COMPONENTS CONTAINS ALL CONNECTED COMPONENTS THAT HAVE LUMINOSITY = PIXEL_LUMINOSITY +- THRESHOLD

		it's inner structure goes as follow:

		connected_components = {
			luminosity : [
				{
				'mask' 		: cv.image # np.array more precisely,
				'area' 		: int
				'centroids'	: tuple
				}
			]
		}

		- luminosity varies from 0 to 255, so connected_components has 256 keys
		- each luminosity contains list of areas that fit the given luminosity threshold
		- this list contains dicts where 'mask' element is the np.array
		"""
		self.connected_components = self.init_comutation_of_connected_components()
		print("Computation finished successfully")
		print("Now get_pixel_theshold_window_mask can be called")

	def get_min_max_area_in_pixels(self, min_area_portion, max_area_portion):
		image_area = self.image.shape[0] * self.image.shape[1]
		min_area_in_pixels = min_area_portion * image_area
		max_area_in_pixels = max_area_portion * image_area
		return min_area_in_pixels, max_area_in_pixels

	def init_comutation_of_connected_components(self):
		all_threshed_images = self.get_all_threshed_images()
		processed_threshed_images = self.morph_process_all_threshed_images(all_threshed_images)
		final_threshed_images = self.get_connected_components_lists_for_all_threshed_images(
			processed_threshed_images)
		return final_threshed_images

	def get_all_threshed_images(self):
		threshed_images = {}
		for pixel_luminosity in range(0, 255 + 1):
			threshed_images[pixel_luminosity] = self.get_threshed_image(pixel_luminosity)
		return threshed_images

	def get_threshed_image(self, pixel_luminosity):
		thresholds = self.get_thresholds(pixel_luminosity)
		threshed_from_bottom_and_top_image = np.array(
			((self.image_l <= thresholds[1]) * (self.image_l >= thresholds[0])) * 255, 
			dtype=np.uint8
			)
		return threshed_from_bottom_and_top_image

	def get_thresholds(self, pixel_luminosity):
		converted_to_255_threshold = int(255 * self.l_threshold_radius)
		left = np.clip(pixel_luminosity - converted_to_255_threshold, 0, 255)
		right = np.clip(pixel_luminosity + converted_to_255_threshold, 0, 255)
		return left, right

	def morph_process_all_threshed_images(self, threshed_images):
		for thresh_value in threshed_images:
			processed_threshed_image = self.morph_process_threshed_image(threshed_images[thresh_value])
			threshed_images[thresh_value] = processed_threshed_image
		return threshed_images

	def morph_process_threshed_image(self, image):
		if not self.kernel_size_erosion == 0:
			image = self.apply_erosion(image)
		if not self.kernel_size_dilation == 0:
			image = self.apply_dilation(image)
		return image

	def apply_circular_morph_operation(self, morphological_operation, image, kernel_size):
		morph_type = cv.MORPH_ELLIPSE
		kernel = cv.getStructuringElement(morph_type, (2*kernel_size + 1, 2*kernel_size+1), 
			(kernel_size, kernel_size))
		morphed_image = morphological_operation(image, kernel)
		return morphed_image

	def apply_erosion(self, image):
		return self.apply_circular_morph_operation(cv.erode, image, self.kernel_size_erosion)

	def apply_dilation(self, image):
		return self.apply_circular_morph_operation(cv.dilate, image, self.kernel_size_dilation)

	def get_connected_components_lists_for_all_threshed_images(self, images):
		all_connected_components = {}
		for thresh_value in images:
			all_connected_components[thresh_value] = self.get_connected_components_list_from_image(
				images[thresh_value]
				)
		return all_connected_components

	def get_connected_components_list_from_image(self, image):
		filtered_connected_components = []
		num_labels, labels, statistics, centroids = self.get_raw_connected_components(image)
		for label in range(num_labels):
			label_area = statistics[label, cv.CC_STAT_AREA]
			if (label_area > self.min_area) and (label_area < self.max_area):
				label_mask = np.array((labels == label) * 255, dtype=np.uint8)
				filtered_connected_components.append({
					'mask' : label_mask,
					'area' : label_area,
					'centroids' : centroids[label]
					})
		return filtered_connected_components

	def get_raw_connected_components(self, image):
		return cv.connectedComponentsWithStats(image, 4, cv.CV_32S)

	#
	#	FUNCTIONS TO BE CALLED OUTSIDE THE CLASS
	#

	# CONNECTED_COMPONENTS CONTAINS ALL CONNECTED COMPONENTS THAT HAVE LUMINOSITY = PIXEL_LUMINOSITY +- THRESHOLD
	# if we want to look at specific pixel, we justs go through all suitable connected areas, until we find
	# the one, that our pixel lies in
	def get_pixel_theshold_window_mask(self, x, y):
		mask = None
		all_pixel_matching_connected_components = self.connected_components[self.get_pixel_luminosity(x, y)]
		for connected_component in all_pixel_matching_connected_components:
			if connected_component['mask'][y, x]:
				mask = connected_component['mask']
		if mask is None:
			print('Warning: connected area withing given threshold does not exist for', "x", x, "y", y)
		return mask

	def get_all_masks_that_has_the_same_l_as_pixel(self, x, y):
		return [cc['mask'] for cc in self.connected_components[self.get_pixel_luminosity(x, y)]]

	def get_pixel_luminosity(self, x, y):
		return self.image_l[y, x]



# Please read __init__ description

if __name__ == "__main__":
	area_selector = ConnectedAreaThresholdSelector(
		# image_path = 'image.png',
		# image_path = '15.png',
		image_path = 'cars.jpg',
		l_threshold_radius = 0.05, 
		min_area_portion = 0.0001, 
		max_area_portion = 0.01,
		kernel_size_erosion = 0,
		kernel_size_dilation = 4
		)

	# here is a good coordinate picker
	# https://summerstyle.github.io/summer/
	random_points = [
	# [470, 480],
	# [215, 747],
	# [460, 260],
	# [572, 765],

	# [615, 459]

	[705, 570],
	[657, 318]
	]

	for point in random_points:
		print(point)
		image_to_draw_on = area_selector.image.copy()
		cv.circle(image_to_draw_on, tuple(point), 10, (0, 255, 0), 3)

		show_image(image_to_draw_on)
		show_image(area_selector.get_pixel_theshold_window_mask(point[0], point[1]))


	if False:
		point_we_want_to_debug_constants_on = [572, 765]

		all_masks_that_has_the_same_l_as_pixel = area_selector.get_all_masks_that_has_the_same_l_as_pixel(
			x = point_we_want_to_debug_constants_on[0],
			y = point_we_want_to_debug_constants_on[1]
			)
		for mask in all_masks_that_has_the_same_l_as_pixel:
			# you can't draw on a mask, so just place cursor on the object you want to debug
			show_image(mask)