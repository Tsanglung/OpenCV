#include <vector>
#include <iostream>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

#define alpha 0.1		// 浮水印強度

using namespace cv;
using namespace std;

void dct512(Mat &);
double PSNR(Mat &, Mat &);
double NC(Mat &, Mat &);
void rotato(Mat &);

int main()
{
	//8bit 灰階載入原始與浮水印影像
	Mat barbara512 = imread("barbara_512x512.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat watermark = imread("watermark.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat dctImage = barbara512.clone(); 									//backup
	Mat wm = watermark.clone();								   			  //backup
	resize(dctImage, dctImage, Size(512, 512));					
	resize(wm, wm, Size(64, 64));
	dctImage.convertTo(dctImage, CV_32F, 1.0 / 255.0);		//32 bit 灰階
	wm.convertTo(wm, CV_32F, 1.0 / 255.0); 						 //32 bit 灰階

	dct512(dctImage);		//DCT 轉換

	// 嵌入浮水印
	Mat block;					//暫存圖區
	for (int i = 0; i < 512; i += 8)
	{
		float W_pixel;	//暫存浮水印 pixel
		for (int j = 0; j < 512; j += 8)
		{
			block = dctImage(Rect(i, j, 8, 8)); //矩形Rect
			W_pixel = (float)wm.at<float>(i / 8, j / 8);
			// 若浮水印 pixel 非白色，該 block 中間顏色區域加入浮水印 pixel
			if (W_pixel < 1) // f'(x, y) = f(x, y) + alpha * w(u, v) 
				block.at<float>(5, 5) = (float)block.at<float>(5, 5) + alpha * W_pixel;	

			idct(block, block);	// DCT 逆轉換
		}
	}

	namedWindow("embed result");										//視窗標題
	moveWindow("embed result",513,0);								 // 設置顯示視窗座標
	imshow("embed result",dctImage);									// 顯示嵌入結果
	dctImage.convertTo(dctImage, CV_8UC1, 255.0);			// 轉成 8bit 灰階
	imwrite("dctImage.bmp", dctImage);								  // 保存嵌入結果
	cout << "PSNR: " << PSNR(barbara512, dctImage) << "%" << endl;

	rotato(dctImage);	//旋轉影像

	// 取出嵌入浮水印, 做DCT後取出差異值並組成浮水印
	Mat block2;											// 暫存圖區，儲存受攻擊之影像
	Mat extractWatermark(Size(64, 64), CV_32F, Scalar(255));	// 用來保存取出的浮水印
	Mat orgImage = barbara512.clone();
	//受攻擊之影像
	Mat attackImg = imread("cut_dctImage.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	resize(orgImage, orgImage, Size(512, 512));					// 原始影像
	resize(attackImg, attackImg, Size(512, 512));				// attack 影像
	orgImage.convertTo(orgImage, CV_32F, 1.0 / 255.0);	
	attackImg.convertTo(attackImg, CV_32F, 1.0 / 255.0);
	dct512(orgImage);
	dct512(attackImg);

	for (int i = 0; i < 512; i += 8)
		for (int j = 0; j < 512; j += 8)
		{
			block = orgImage(Rect(i, j, 8, 8));						//原始影像 block
			block2 = attackImg(Rect(i, j, 8, 8));	  //被攻擊影像 block

			if (block.at<float>(5, 5) != block2.at<float>(5, 5))  // w(u, v) = (f''(x, y) - f(x, y)) / alpha
				extractWatermark.at<float>(i / 8, j / 8) = (block2.at<float>(5, 5) - block.at<float>(5, 5)) / alpha; 
		}

	namedWindow("get watermark");
	moveWindow("get watermark",0,635);
	imshow("get watermark",extractWatermark);
	extractWatermark.convertTo(extractWatermark, CV_8UC1, 255.0);		 // 將影像轉成 8bit 灰階		
    imwrite("extractWatermark.bmp", extractWatermark); 					//保存取出的浮水印影像

	//NC
	resize(extractWatermark, extractWatermark, Size(64, 64));
	cout << "NC: " << NC(wm, extractWatermark) * 100 <<"%"<< endl;			
	waitKey(0);
	destroyAllWindows();
	return 0;
}

double PSNR(Mat & I1, Mat & I2) //PSNR 
{  //I1 = barbara_512x512.bmp
	//I2 = embed result
	I1.convertTo(I1,CV_32FC1); 			// 轉成 32bit 灰階
	I2.convertTo(I2,CV_32FC1);

	float MSE = 0.0;

	for(int i = 0;i < I1.rows; i++)
		for(int j = 0;j < I1.cols ; j++)
			MSE += pow(I1.at<float>(i,j) - I2.at<float>(i,j),2);

	MSE /= I1.rows * I1.cols; //512 x 512

	I1.convertTo(I1, CV_8UC1);			// 還原 8bit 灰階
	I2.convertTo(I2, CV_8UC1);

	double psnr = (10 * log10((255 * 255) / MSE));
	return isinf(psnr) ? 100 : psnr;
}

void dct512(Mat & dct_I) 
{	//將512*512的圖做DCT
	Mat block;
	// 將影像分割成8x8的圖區並個別執行DCT轉換
	for (int i = 0; i < 512; i += 8)
		for (int j = 0; j < 512; j += 8)
		{
			block = dct_I(Rect(i, j, 8, 8));
			dct(block, block);
		}
}


double NC(Mat &I1, Mat &I2) 
{ //計算 NC 值
	// I1 = watermark.bmp
	// I2 = extract watermark
	I1.convertTo(I1,CV_32FC1); 			// 轉成 32bit 灰階
	I2.convertTo(I2,CV_32FC1);

	float up = 0.0,down = 0.0,down2 = 0.0;
	for (int i = 0; i < I1.rows; i++)
		for (int j = 0; j < I1.cols; j++) {
			up += I1.at<float>(i, j) * I2.at<float>(i, j);
			down += pow(I1.at<float>(i,j),2);
			down2 += pow(I2.at<float>(i,j),2);
		}
	down = sqrt(down *down2);
	//down = I1.rows * I1.cols;
	cout<<up<<" "<<down<<endl;
	return up / down;
}

void rotato(Mat & I)	//旋轉影像
{	//正:逆時針旋轉,負:順時針旋轉
	float angle = -2; 																				//旋轉 angle 度
	Point2f center((I.cols - 1) / 2.0,(I.rows - 1) / 2.0); 							//取得影像中心座標
	Mat rotation_matrix = getRotationMatrix2D(center,angle,1.0);    //得到 2D 旋轉矩陣
	Mat rotated_image;
	warpAffine(I,rotated_image,rotation_matrix,I.size()); 					//warpAffine旋轉影像
	imshow("Rotated Image",rotated_image);										
	imwrite("rotated_dctImage.bmp",rotated_image);						   //儲存
}
