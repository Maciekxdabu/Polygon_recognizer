#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

cv::Vec3f ConvertBGRtoHSV(cv::Vec3b& bgr)
{
	//normalize the bgr values
	float b = bgr[0] / 255.0;
	float g = bgr[1] / 255.0;
	float r = bgr[2] / 255.0;

	//find maximum and minimal value and their delta
	float maxValue = std::max({ b, g, r });
	float minValue = std::min({ b, g, r });
	float delta = maxValue - minValue;

	//calculate v (0-1)
	float v = maxValue;

	//calculate s (0-1)
	float s = (maxValue == 0) ? 0 : delta / maxValue;

	//calculate h (0-360)
	float h = 0.0;
	if (delta != 0) {
		if (maxValue == r) {
			h = 60 * (fmod(((g - b) / delta), 6));
		}
		else if (maxValue == g) {
			h = 60 * (((b - r) / delta) + 2);
		}
		else if (maxValue == b) {
			h = 60 * (((r - g) / delta) + 4);
		}
		if (h < 0) {
			h += 360;
		}
	}

	//scaling the values to fit (0-100) for S and V
	v *= 100;
	s *= 100;

	return cv::Vec3f(h, s, v);
}

bool CheckColor(cv::Vec3b checkedColor)
{
	//Remainder: the order is: GRB (Green, Red, Blue)
	//Official colors: #fab40a (pomarañczowy) #212121 (odcieñ czerni w tle)
	//RGB:   R: 250    G: 180    B: 10
	//RGB:   R: 33     G: 33     B: 33

	//std::vector<uchar> color = {10, 180, 250};//BGR
	std::vector<float> color = {43, 96, 98 };//HSV (0-360, 0-100, 0-100)
	std::vector<float> allowedDelta = { 50, 40, 40 };//max difference

	//convert given color to HSV
	cv::Vec3f hsv = ConvertBGRtoHSV(checkedColor);

	for (int i = 0; i < 3; i++)
	{
		if (abs((int)hsv[i] - color[i]) >= allowedDelta[i])
			return false;
	}

	return true;
}

//extracts shapes for the searched symbol (returns a new image)
cv::Mat ExtractShapes(cv::Mat& I)
{
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC1);//8-bitowe, unsigned, 3 sztuki
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		cv::Mat_<uchar> result = res;//grayscale
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				if (CheckColor(_I(i, j)))
					result(i, j) = 255;
				else
					result(i, j) = 0;
			}
		res = result;
		break;
	}

	return res;
}

//Displays the gicen image with some scaling to fit in the screen
void DisplayImage(cv::Mat& I, std::string windowName)
{
	cv::namedWindow(windowName, cv::WINDOW_NORMAL);
	cv::MatSize size = I.size;
	float ratio = 1.0f * size[0] / size[1];
	if (ratio < 1.0)
		cv::resizeWindow(windowName, 800, 800 * ratio);
	else
		cv::resizeWindow(windowName, 500, 500 * ratio);
	cv::imshow(windowName, I);
}

int main()
{
	//get all images paths from the input folder
	std::vector<cv::String> imagesPaths;
	cv::glob("zdjecia", imagesPaths, false);

	for (int i = 0; i < MIN(imagesPaths.size(), 10); i++)
	{
		//Load the image (check if empty)
		cv::Mat img = cv::imread(imagesPaths[i]);
		if (img.empty())
		{
			std::cout << "Image could not be loaded" << std::endl;
			return 0;
		}

		//Processing
		cv::Mat step1 = ExtractShapes(img);
		//---TODO---

		//Output the image to the output folder
		//---TODO---

		//==========DEBUG==========
		std::cout << "Dimensions: " << std::to_string(img.size[0]) + " x " + std::to_string(img.size[1]) << std::endl;
		std::cout << "Dimensions: " << std::to_string(step1.size[0]) + " x " + std::to_string(step1.size[1]) << std::endl;

		//display image
		DisplayImage(img, "Image " + std::to_string(i));
		DisplayImage(step1, "Image " + std::to_string(i) + "   Step: 1");

		//wait for input and close all windows to close
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	
	return 0;
}

/* INFO
	Available OpenCV methods:
	- cv::imread - czytanie obrazu z pliku
	- cv::imwrite - zapis obrazu do pliku
	- cv::Mat i cv::_Mat - konstruktory
	- cv::imshow - wyœwietlanie obrazu
	- cv::waitkey - czekanie na klawisz
	- std::cout - wypisywanie informacji na konsolê
	*/