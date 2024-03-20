#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;

void convertToGrayscale(Mat& input, Mat& output, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        uchar* inputPtr = input.ptr<uchar>(i);
        uchar* outputPtr = output.ptr<uchar>(i);

        for (int j = 0; j < input.cols; ++j) {
            int blue = inputPtr[3 * j];
            int green = inputPtr[3 * j + 1];
            int red = inputPtr[3 * j + 2];

            int gray = (red + green + blue) / 3;

            outputPtr[j] = gray;
        }
    }
}

void applySepia(Mat& input, Mat& output, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        uchar* inputPtr = input.ptr<uchar>(i);
        uchar* outputPtr = output.ptr<uchar>(i);

        for (int j = 0; j < input.cols; ++j) {
            int blue = inputPtr[3 * j];
            int green = inputPtr[3 * j + 1];
            int red = inputPtr[3 * j + 2];

            int sepiaBlue = (int)(0.272f * red + 0.534f * green + 0.131f * blue);
            int sepiaGreen = (int)(0.349f * red + 0.686f * green + 0.168f * blue);
            int sepiaRed = (int)(0.393f * red + 0.769f * green + 0.189f * blue);

            sepiaBlue = min(255, sepiaBlue);
            sepiaGreen = min(255, sepiaGreen);
            sepiaRed = min(255, sepiaRed);

            outputPtr[3 * j] = (uchar)sepiaBlue;
            outputPtr[3 * j + 1] = (uchar)sepiaGreen;
            outputPtr[3 * j + 2] = (uchar)sepiaRed;
        }
    }
}

void applyNegative(Mat& input, Mat& output, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        uchar* inputPtr = input.ptr<uchar>(i);
        uchar* outputPtr = output.ptr<uchar>(i);

        for (int j = 0; j < input.cols; ++j) {
            outputPtr[3 * j] = 255 - inputPtr[3 * j];         // Blue
            outputPtr[3 * j + 1] = 255 - inputPtr[3 * j + 1]; // Green
            outputPtr[3 * j + 2] = 255 - inputPtr[3 * j + 2]; // Red
        }
    }
}

void applySobel(Mat& input, Mat& output, int startRow, int endRow) {
    Mat sobelX = (Mat_<int>(3, 3) << 1, 0, -1,
        2, 0, -2,
        1, 0, -1);

    Mat sobelY = (Mat_<int>(3, 3) << 1, 2, 1,
        0, 0, 0,
        -1, -2, -1);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            int sumX = 0;
            int sumY = 0;

            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int rowIndex = i + k;
                    int colIndex = j + l;

                    if (rowIndex >= 0 && rowIndex < input.rows && colIndex >= 0 && colIndex < input.cols) {
                        sumX += input.at<uchar>(rowIndex, colIndex) * sobelX.at<int>(k + 1, l + 1);
                        sumY += input.at<uchar>(rowIndex, colIndex) * sobelY.at<int>(k + 1, l + 1);
                    }
                }
            }

            int gradient = sqrt(sumX * sumX + sumY * sumY);
            output.at<uchar>(i, j) = 255 - saturate_cast<uchar>(gradient);
        }
    }
}

void invertColors(Mat& input, Mat& output) {
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            if (input.at<uchar>(i, j) > 127) { 
                output.at<uchar>(i, j) = 255 - input.at<uchar>(i, j); 
            }
            else {
                output.at<uchar>(i, j) = input.at<uchar>(i, j); 

            }
        }
    }
}

int main() {
    Mat image = imread("D:/mount.jpg");

    if (image.empty()) {
        cerr << "Error" << endl;
        return 1;
    }

    Mat resizedImage;
    resize(image, resizedImage, Size(), 0.25, 0.25);

    Mat grayscale(resizedImage.rows, resizedImage.cols, CV_8UC1);
    Mat sepia(resizedImage.rows, resizedImage.cols, CV_8UC3);
    Mat negative(resizedImage.rows, resizedImage.cols, CV_8UC3);
    Mat sobel(resizedImage.rows, resizedImage.cols, CV_8UC1);

    const int numThreads = thread::hardware_concurrency();
    vector<thread*> threads(numThreads);
    int rowsPerThread = resizedImage.rows / numThreads;
    int startRow = 0;
    int endRow = 0;


    for (int i = 0; i < numThreads; ++i) {
        startRow = i * rowsPerThread;
        endRow = (i == numThreads - 1) ? resizedImage.rows : startRow + rowsPerThread;
        threads[i] = new thread(convertToGrayscale, ref(resizedImage), ref(grayscale), startRow, endRow);
    }

    for (int i = 0; i < numThreads; ++i) {
        threads[i]->join();
        delete threads[i];
    }


    for (int i = 0; i < numThreads; ++i) {
        startRow = i * rowsPerThread;
        endRow = (i == numThreads - 1) ? resizedImage.rows : startRow + rowsPerThread;
        threads[i] = new thread(applySepia, ref(resizedImage), ref(sepia), startRow, endRow);
    }

    for (int i = 0; i < numThreads; ++i) {
        threads[i]->join();
        delete threads[i];
    }


    for (int i = 0; i < numThreads; ++i) {
        startRow = i * rowsPerThread;
        endRow = (i == numThreads - 1) ? resizedImage.rows : startRow + rowsPerThread;
        threads[i] = new thread(applyNegative, ref(resizedImage), ref(negative), startRow, endRow);
    }

    for (int i = 0; i < numThreads; ++i) {
        threads[i]->join();
        delete threads[i];
    }

    for (int i = 0; i < numThreads; ++i) {
        startRow = i * rowsPerThread;
        endRow = (i == numThreads - 1) ? resizedImage.rows : startRow + rowsPerThread;
        threads[i] = new thread(applySobel, ref(grayscale), ref(sobel), startRow, endRow);
    }

    for (int i = 0; i < numThreads; ++i) {
        threads[i]->join();
        delete threads[i];
    }




    imshow("Original", resizedImage);
    imshow("Grayscale", grayscale);
    imshow("Sepia", sepia);
    imshow("Negative", negative);
    imshow("Inverted Sobel", sobel);
    waitKey(0);

    return 0;
}
