#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <stack>
#include <cmath>

# define M_PI 3.14159265358979323846 /* pi */

// START CONFIG
int minPixelCount = 100; //minimum number of pixels for the shape to be considered valid for analysis (to filter amount too small shapes)
// END CONFIG

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

//method for clearing binary images (sets all pixels to '0')
void ClearImage(cv::Mat_<uchar>& img)
{
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
			img(i, j) = 0;
}

#pragma region Filtering_methods

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
	std::vector<float> color = { 43, 96, 98 };//HSV (0-360, 0-100, 0-100)
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

//extracts shapes for the searched logo (returns a new black and white image)
cv::Mat ExtractShapes(cv::Mat& I)
{
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC1);//8-bitowe, unsigned, 1 sztuka
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

#pragma endregion

#pragma region Exracting_shapes_methods

struct ShapeImage
{
	cv::Mat_<uchar> shape;
	cv::Rect boundingBox;
	ShapeImage(cv::Mat_<uchar> shape, cv::Rect boundingBox) : shape(shape), boundingBox(boundingBox) {}
};

struct PixelCoord
{
	int x, y;
	PixelCoord(int x, int y) : x(x), y(y) {}
};

//recursive method for probing the pixels of the shape (returns the number of pixels in the givenShape)
int ProbeShape(cv::Mat_<uchar>& searchedImage, cv::Mat_<uchar>& resultImage, bool** checkedPixels, cv::Rect& boundingBox, int x, int y)
{
	/*//mark pixel as checked
	checkedPixels[x][y] = true;
	int suma = 0;

	//check if pixel is white
	if ((*searchedImage)(x, y) == 255)
	{
		//add it to the result
		(*resultImage)(x, y) = 255;
		suma++;

		//update bounding box if needed
		if (x < (*boundingBox).x)//left
		{
			(*boundingBox).x--;
			(*boundingBox).width++;
		}
		else if ((*boundingBox).x + (*boundingBox).width < x)//right
			(*boundingBox).width++;
		if (y < (*boundingBox).y)//up
		{
			(*boundingBox).y--;
			(*boundingBox).height++;
		}
		else if ((*boundingBox).y + (*boundingBox).height < y)//down
			(*boundingBox).height++;
	}

	//check the other directions if not yet checked

	if (x > 0 && checkedPixels[x - 1][y] == false)//left
		suma += ProbeShape(searchedImage, resultImage, checkedPixels, boundingBox, x - 1, y);

	if (y > 0 && checkedPixels[x][y - 1] == false)//up
		suma += ProbeShape(searchedImage, resultImage, checkedPixels, boundingBox, x, y - 1);

	if (x < (*searchedImage).size[0] - 1 && checkedPixels[x + 1][y] == false)//right
		suma += ProbeShape(searchedImage, resultImage, checkedPixels, boundingBox, x + 1, y);

	if (y < (*searchedImage).size[1] - 1 && checkedPixels[x][y + 1] == false)//down
		suma += ProbeShape(searchedImage, resultImage, checkedPixels, boundingBox, x, y + 1);

	return suma;*/

	int suma = 0;

	// Directions for moving up, down, left, and right
	std::vector<PixelCoord> directions = { PixelCoord(-1, 0), PixelCoord(1, 0), PixelCoord(0, -1), PixelCoord(0, 1) };

	// Initialize the stack
	std::stack<PixelCoord> stack;
	stack.push(PixelCoord(x, y));

	// Initialize the extremas (for bounding box)
	int minX = x, maxX = x, minY = y, maxY = y;

	while (!stack.empty())
	{
		PixelCoord p = stack.top();
		stack.pop();

		// Mark the pixel as checked
		checkedPixels[p.y][p.x] = true;

		// Mark the pixel in the result image
		resultImage(p.y, p.x) = 255;

		//increase the pixel count
		suma++;

		// Update the extremas
		minX = MIN(minX, p.x);
		maxX = MAX(maxX, p.x);
		minY = MIN(minY, p.y);
		maxY = MAX(maxY, p.y);

		// Explore neighbors
		for (const PixelCoord& dir : directions)
		{
			int newX = p.x + dir.x;
			int newY = p.y + dir.y;

			// Check if the new position is within bounds and is a white pixel
			if (newX >= 0 && newX < searchedImage.cols && newY >= 0 && newY < searchedImage.rows && searchedImage(newY, newX) == 255 && !checkedPixels[newY][newX])
			{
				stack.push(PixelCoord(newX, newY));
			}
		}
	}

	// Set the bounding box based on found extremas
	boundingBox = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);

	if (suma >= minPixelCount)
		std::cout << "Finished probing a shape, it has: " << suma << " pixels" << std::endl;

	return suma;
}

//remove a shape presented on the second image from the base image based on the boundingBox
void RemoveShape(cv::Mat_<uchar>& baseImage, cv::Mat_<uchar>& shapeToRemove, cv::Rect boundingBox)
{
	//caching boundary values
	int x2 = boundingBox.x + boundingBox.width;
	int y2 = boundingBox.y + boundingBox.height;

	//main loop
	for (int i = boundingBox.y; i < y2; i++)
		for (int j = boundingBox.x; j < x2; j++)
		{
			if (shapeToRemove(i, j) == 255)
				baseImage(i, j) = 0;
		}
}

//separates the unconnected shapes into different matrices (cv::Mat's)
std::vector<ShapeImage> SeparateShapes(cv::Mat& I)
{
	std::vector<ShapeImage> results;

	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC1);//8-bitowe, unsigned, 1 sztuka
	switch (I.channels()) {
	case 1:
		cv::Mat_<uchar> _I = I;
			
		bool** checkedPixels = new bool* [_I.size[0]];
		for (int i = 0; i < _I.size[0]; i++)
		{
			checkedPixels[i] = new bool[_I.size[1]];
			for (int j = 0; j < _I.size[1]; j++)
				checkedPixels[i][j] = false;
		}

		//search for white pixels in the image (and by that through)
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				//mark pixel as checked
				checkedPixels[i][j] = true;

				if (_I(i, j) == 255)//encountered a white pixel
				{
					//initialize values
					cv::Rect boundingBox;
					cv::Mat_<uchar> resultShape = res.clone();//grayscale
					ClearImage(resultShape);
					//TODO ----- clear empty image

					//do shape search
					int shapeSize = ProbeShape(_I, resultShape, checkedPixels, boundingBox, j, i);

					//remove the found shape from main image
					RemoveShape(_I, resultShape, boundingBox);

					//add found shape to output if big enough
					if (shapeSize >= minPixelCount)
					{
						ShapeImage newShape(resultShape, boundingBox);
						results.push_back(newShape);
					}
				}
			}
		//when through all the pixels, and by that through all the shapes
		
		break;
	}

	return results;
}

//returns a new image with drawn bounding boxes from a list ===== NYI =====
cv::Mat DrawBoundingBoxes(cv::Mat& I, std::vector<cv::Rect> boundingBoxes)
{
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC1);//8-bitowe, unsigned, 1 sztuka
	switch (I.channels()) {
	case 1:
		cv::Mat_<cv::Vec3b> _I = I;
		cv::Mat_<uchar> result = res;//grayscale
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
			{
				
			}
		res = result;
		break;
	}

	return res;
}

#pragma endregion


#pragma region Coefficients_methods

//pole figury
float S_Coeff(cv::Mat& I)
{
	CV_Assert(I.depth() != sizeof(uchar));
	float S = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if (_I(i, j)[0] == 0)
					S++;
			}
		break;
	}
	return S;
}

//obwód figury
float L_Coeff(cv::Mat& I)
{
	CV_Assert(I.depth() != sizeof(uchar));
	float L = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if (_I(i, j)[0] == 0)//we have a black pixel
				{
					if ((i + 1 < I.rows && _I(i + 1, j)[0] != 0) ||
						(i - 1 >= 0 && _I(i - 1, j)[0] != 0) ||
						(j + 1 < I.cols && _I(i, j + 1)[0] != 0) ||
						(j - 1 >= 0 && _I(i, j - 1)[0] != 0) ||
						(_I(i + 1, j + 1)[0] != 0) ||
						(_I(i + 1, j - 1)[0] != 0) ||
						(_I(i - 1, j + 1)[0] != 0) ||
						(_I(i - 1, j - 1)[0] != 0))
						L++;
				}
			}
		break;
	}
	return L;
}

float W3_Coeff(float S, float L)
{
	float W3 = L / (2 * sqrt(M_PI * S)) - 1;

	return W3;
}

float normalMoment(cv::Mat& I, int p, int q)
{
	CV_Assert(I.depth() != sizeof(uchar));
	float moment = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if (_I(i, j)[0] == 0)
					moment += pow(i, p) * pow(j, q);
			}
		break;
	}
	return moment;
}

float CentralMoment(cv::Mat& I, int p, int q)
{
	float centralMoment = 0;
	//2,0  0,2  1,1
	if (p == 1 && q == 1)//1,1
	{
		centralMoment = normalMoment(I, 1, 1) - (normalMoment(I, 1, 0) * normalMoment(I, 0, 1)) / normalMoment(I, 0, 0);
	}
	else if (p == 2 && q == 0)//2,0
	{
		centralMoment = normalMoment(I, 2, 0) - pow(normalMoment(I, 1, 0), 2) / normalMoment(I, 0, 0);
	}
	else if (p == 0 && q == 2)//0,2
	{
		centralMoment = normalMoment(I, 0, 2) - pow(normalMoment(I, 0, 1), 2) / normalMoment(I, 0, 0);
	}

	return centralMoment;
}

float M1_Coeff(cv::Mat& I)
{
	float M1 = (CentralMoment(I, 2, 0) + CentralMoment(I, 0, 2)) / pow(normalMoment(I, 0, 0), 2);

	return M1;
}

float M7_Coeff(cv::Mat& I)
{
	float M7 = (CentralMoment(I, 2, 0) * CentralMoment(I, 0, 2) - pow(CentralMoment(I, 1, 1), 2)) / pow(normalMoment(I, 0, 0), 4);

	return M7;
}

#pragma endregion

// ======================= MAIN =======================
int main()
{
	std::string inputFolderName = "zdjecia";
	std::string outputFolderName = "output";

	//get all images paths from the input folder
	std::vector<cv::String> imagesPaths;
	cv::glob(inputFolderName, imagesPaths, false);

	for (int i = 0; i < MIN(imagesPaths.size(), 10); i++)
	{
		//==========DEBUG==========
		std::cout << "Image: " << imagesPaths[i] << std::endl;

		//Load the image (check if empty)
		cv::Mat img = cv::imread(imagesPaths[i]);
		if (img.empty())
		{
			std::cout << "Image could not be loaded" << std::endl;
			return 0;
		}
		std::string outputPath = outputFolderName + "/" + imagesPaths[i].substr(inputFolderName.size());//prepare the path for the file to output results

		DisplayImage(img, "Image " + std::to_string(i));

		//==========DEBUG==========
		std::cout << "Image loaded: dimensions: " << std::to_string(img.size[0]) + " x " + std::to_string(img.size[1]) << std::endl;
		std::cout << "Processing..." << std::endl;

		//Processing - filtering
		cv::Mat step1 = ExtractShapes(img);//extract shapes into a single black and white picture
		cv::imwrite(outputPath + " - step 1.png", step1);
		DisplayImage(step1, "Image " + std::to_string(i) + "   Step: 1");

		//Processing - separating shapes
		std::vector<ShapeImage> separatedShapes = SeparateShapes(step1);
		std::cout << "Amount of individual shapes found: " << separatedShapes.size() << std::endl;
		for (int j=0; j<separatedShapes.size(); j++)
		{
			DisplayImage(separatedShapes[j].shape, "Image " + std::to_string(i) + "   Step: 2 Shape: " + std::to_string(j+1));
			std::cout << "Shape " << std::to_string(j + 1) << " bounding box:" << 
				" x: " << std::to_string(separatedShapes[j].boundingBox.x) << " y: " << std::to_string(separatedShapes[j].boundingBox.y) << 
				" width: " << std::to_string(separatedShapes[j].boundingBox.width) << " height: " << std::to_string(separatedShapes[j].boundingBox.height) << std::endl;
		}
		
		//Processing - Check for coefficients
		//TODO

		//mark the found shapes on the original image
		//TODO
		

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