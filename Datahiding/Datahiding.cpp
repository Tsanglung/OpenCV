#include <iostream>
#include <vector>                           //向量
#include <bitset>                            //位元
#include <random>                        //亂數取值
#include <numeric>                       //使用 iota 函數
#include <direct.h>                        //mkdir 創建資料夾
#include <opencv2/opencv.hpp>   //opencv 影像處理
#include <opencv2/core/utils/logger.hpp>

using namespace std;
using namespace cv;

const int MOD = 251; //mod 值

void Coefficient(int p,int &S, int &a, int &b, int &c) 
{    // 建構多項式係數 a、b、c 並嵌入秘密資訊
    if (p < 251)    //case 1
    { // pixel < 251
        a = (S >> 8);   //(S1 S2 S3 S4 S5 S6 S7)
        b = p;
        c = bitset<7>(S >>1).to_ulong(); // S8~S14
        if (a + 128 < 251 && c + 128 < 251) 
        { // 公式 2、3
            if (bitset<1>(S).to_ulong() == 1)  a += 128;// S_15 = 1
            if (bitset<1>(S).to_ulong() == 0)  c += 128;// S_15 = 0
        }
    } 
    else if (p >= 251) 
    {   //case 2
        a = (2 & 0b11) << 6 | (S >> 13) << 4 | (p >> 4) << 0;   // (1 0 S1 S2 p7 p6 p5 p4)
        b = bitset<7>(S >> 4).to_ulong();   //S5 ~ S11
        c = (2 & 0b11) << 6 | (S >> 11) << 4 | bitset<4>(p).to_ulong(); //(1 0 S2 S3 p3 p2 p1 p0)
    }
}

int ModInverse(int a , int m) // 使用歐基里德延伸演算法求模反元素
{ // 找出 x ，使得 (a * x) mod m = 1
    int m_tmp = m, t, q;
    int t0 = 0, t1 = 1;
    while (a > 1) 
    { // 使用歐基里德延伸演算法
        q = a / m;  //商
        t = m;
        m = a % m;  //餘數更新為新的 m
        a = t;  //原來的 m 變為新的 a
        t = t0; t0 = t1 - q * t0; t1 = t;
    }
    if (t1 < 0)  t1 += m_tmp; // 確保結果為正數(+ 模數)
    return t1;
}

void Lagrange(const vector<int>& X , const vector<int>& C , int &a , int& b , int& c) 
{   // 計算多項式係數
    a = 0, b = 0, c = 0;
    for (int t = 0; t < 3; ++t) 
    {
        int L_t = C[t];
        for (int i = 0; i < 3; ++i) 
            if (i != t)  //L_t * (X[t] - X[i]) 的模反元素。找出 x ，使得 ((X[t] - X[i] + MOD) % MOD) * x) mod MOD = 1
                L_t = L_t * ModInverse((X[t] - X[i] + MOD) % MOD, MOD) % MOD;

        int current_a = L_t; //x^2 的係數 
        // ((-X[(t + 1) % 3] - X[(t + 2) % 3] + 2 * MOD) % MOD)  計算此兩點的負和
        int current_b = L_t * ((-X[(t + 1) % 3] - X[(t + 2) % 3] + 2 * MOD) % MOD) % MOD; //x 係數
        int current_c = L_t * ((X[(t + 1) % 3] * X[(t + 2) % 3]) % MOD) % MOD; //常數係數
        a = (a + current_a) % MOD;
        b = (b + current_b) % MOD;
        c = (c + current_c) % MOD;  
    }
}


void Extract(const vector<int>& X , const vector<int>& C , int &P , bitset<15> &S) 
{   // 解密、復原提取影像 pixel
    int a, b, c;
    Lagrange(X, C, a, b, c);    // "F(x) = " << a << "x^2 + " << b << "x + " << c << " (mod 251)" 
    int a_count = 0,c_count = 0,tmp_a = a,tmp_c = c;
    while(tmp_a > 0 || tmp_c > 0)
    {   //計算a、c的位元數
        if(tmp_a > 0)
        {
            a_count++;
            tmp_a >>= 1;
        }
        if(tmp_c > 0)
        {
            c_count++;
            tmp_c >>= 1;
        }
    }
    if(a_count < 8) {a = bitset<7>(a).to_ulong();a_count = 7;}
    else {a = bitset<8>(a).to_ulong();c_count = 8;}
    if(c_count < 8) {c = bitset<7>(c).to_ullong();c_count = 7;}
    else {c = bitset<8>(c).to_ulong();c_count = 8;}

    int MSB_a = (a >> (a_count - 1));// a7 a6 a5 a4 a3 a2 a1x
    int MSB_c = (c >> (c_count - 1));

    if(MSB_a == 1 && MSB_c == 1)
        P = bitset<8>((a << 4) | c).to_ulong(); //(a3 a2 a1 a0 c3 c2 c1 c0)_2 to int
    else 
        P = b;

    if(MSB_a == 0 && MSB_c == 0)
    {   //(a6 a5 a4 a3 a2 a1 a0 c6 c5 c4 c3 c2 c1 c0)_2 to int
        int tmp = bitset<14>((a<<7) | c).to_ulong();   
        S = bitset<15>(tmp);
    }
    else if(MSB_a == 1 && MSB_c == 0)
        S = bitset<15>((a << 8) | (c << 1) | ((2 >> 1) & 1));   //(a6 a5 a4 a3 a2 a1 a0 c6 c5 c4 c3 c2 c1 c0 1)_2
    else if(MSB_a == 0 && MSB_c == 1)
        S = bitset<15>((a <<8) | (c << 1) | ((2 >> 2) & 1));    //(a6 a5 a4 a3 a2 a1 a0 c6 c5 c4 c3 c2 c1 c0 0)_2
    else if(MSB_a == 1 && MSB_c == 1) 
    {   //(a5 a4 c5 c4 b6 b5 b4 b3 b2 b1 b0)_2
        int tmp = bitset<11>((a << 9) | (c << 7) | b).to_ulong();  
        S = bitset<15>(tmp);
    }
}

vector<int> SelectIndexes(int total, int select) 
{   // 隨機選取三個不同的分享影像索引
    vector<int> indices(total);
    iota(indices.begin(), indices.end(), 0); //產生連續的值
    //利用 mt19937 標準的隨機數生成器（Mersenne Twister），將 indices 內的元素隨機打亂
    shuffle(indices.begin(), indices.end(), mt19937(random_device()()));
    return vector<int>(indices.begin(), indices.begin() + select); //回傳 3 個影像索引值
}

double PSNR(Mat & I1,Mat& I2)
{
    I1.convertTo(I1,CV_32FC1); 
    I2.convertTo(I2,CV_32FC1);
    float MSE = 0.0;
    for(int i = 0; i  < I1.rows; i++)
        for(int j = 0; j < I1.cols; j++)
            MSE += pow(I1.at<float>(i,j) - I2.at<float>(i,j) , 2);

    MSE /= I1.rows * I1.cols;

    I1.convertTo(I1,CV_8UC1);
    I2.convertTo(I2,CV_8UC1);

    double psnr = (10 * log10((255 * 255) / MSE));
    return isinf(psnr) ? 100 : psnr;
}

int main() {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);
    string Path = "Image Source/*.bmp"; // 存有影像的資料夾路徑
    vector<Mat> Images;                         // 存放讀取影像
    vector<string> ImageName;              //存放讀取影像的名稱
    glob(Path, ImageName, false);                        // 將資料夾路徑下的影像名稱存到 ImageName
    size_t ImagesCount = ImageName.size();      // 影像數量

    for (size_t i = 0; i < ImagesCount; i++)          // 讀取灰階影像
        Images.push_back(imread(ImageName[i], IMREAD_GRAYSCALE));

    for (size_t I_Count = 0; I_Count < ImagesCount; I_Count++)
    {   //從讀取到的第 I_Count 張影像開始處理，直到所有影像處理完成。
        vector<int> X = {60, 80, 100, 40, 50,200};                                                          // 金鑰
        vector<Mat> C;                                                                                              // 初始化分享影像
        int a, b, c, p,Embedded_bits = 0;
        for (int C_Index = 0; C_Index < X.size(); C_Index++) 
        {   //根據金鑰數量決定分享影像數量
            Mat Ct = Mat::zeros(Images[I_Count].size(), CV_8UC1);
            C.push_back(Ct);
        }

        for (int i = 0; i < Images[I_Count].rows; ++i) // 對每個像素值加密、嵌入隱藏資料
            for (int j = 0; j < Images[I_Count].cols; ++j) 
            {   // 加密與嵌入 step1 
                p = Images[I_Count].at<uchar>(i, j); // 提取像素值
                if(p < 251) Embedded_bits+=15;//有嵌入隱藏資訊的像素值
                //else if(p >= 251) cout<<I_Count<<" "<<(i,j)<<" "<<p<<endl;
                int S;
                if ((j + 1) % 2 == 1) S = 4094;
                else if ((j + 1) % 2 == 0) S = 3237;
                Coefficient(p, S, a, b, c);//得到 a、b、c 值
                // y ^ t (i,j) = a X_t^2 + b X_t + c (mod 251)
                for (int t = 0; t < X.size(); t++)               // 加密與嵌入 step 2，獲得 n 個加密後的像素值
                    C[t].at<uchar>(i, j) = (a * X[t] * X[t] + b * X[t] + c) % MOD;
            } 

        size_t pos1 = ImageName[I_Count].find_last_of("\\");                //字串路徑名稱 "\" 的位置
        size_t pos2 = ImageName[I_Count].find_last_of(".");                 //字串路徑名稱 "." 的位置
        size_t pos3 = ImageName[I_Count].find_last_of("_" , pos2);     //字串路徑名稱在 "." 之前的最後一個 "_" 的位置
        string FileName = ImageName[I_Count].substr(pos1 + 1 , pos3 - pos1 - 1);    //取出子字串

        mkdir(FileName.c_str());                                                              //創建資料夾
        for (int C_Count = 0; C_Count < C.size(); C_Count++)   // 保存分享影像 C ^ t
            imwrite(FileName + "\\" + "shared " + FileName + " " + to_string(C_Count + 1) + ".bmp", C[C_Count]);

        cout<<FileName + " Embedded Bits : " << Embedded_bits  <<endl;//嵌入量
        cout<<FileName + " Embedded Rate : " << (double)(Embedded_bits ) / (Images[I_Count].rows * Images[I_Count].cols * 3)<<endl;//嵌入率
        
        Mat M = Mat::zeros(Images[I_Count].size(), CV_8UC1); // 提取還原影像
        bitset<15> Si;
        vector<bitset<15>> Data;
        vector<int> RIndexes = SelectIndexes(X.size(), 3); // 隨機選取三個不同的分享影像索引
        int P;  //pixel
        cout<<"Select Shared Image Index : "<<RIndexes[0] + 1<<" "<<RIndexes[1] + 1<<" "<<RIndexes[2] + 1<<endl; //3 個影像索引
        for (int i = 0; i < Images[I_Count].rows; ++i)      //對每一像素值進行解密和提取原始影像像素值
            for (int j = 0; j < Images[I_Count].cols; ++j) 
            {
                // 取得任意 3 張分享影像像素值
                vector<int> C_p = {C[RIndexes[0]].at<uchar>(i, j), // RIndexes 值不重複
                                                C[RIndexes[1]].at<uchar>(i, j), 
                                                C[RIndexes[2]].at<uchar>(i, j)}; 
                //任意 3 張分享影像像素值與對應的 X 金鑰
                Extract({X[RIndexes[0]] , X[RIndexes[1]] , X[RIndexes[2]]} , C_p , P , Si);
                M.at<uchar>(i, j) = P;      // 提取像素值
                Data.push_back(Si);        // 提取隱藏資料 S
            }

        cout <<"Hiding Data_(2) : "<< Data[0] << " " << Data[1] << endl;//2進位的隱藏資料
        cout <<"Hiding Data_(10) : " << Data[0].to_ulong() << " " << Data[1].to_ulong() << endl;//10進位的隱藏資料
        imwrite(FileName + "\\" + "get " + FileName + ".bmp", M);
        cout<<"PSNR of " + FileName + "_512x512.bmp and Recovered " + FileName + ".bmp = ";
        cout<<PSNR(Images[I_Count],M)<<"%"<<endl<<endl;
        imshow("Original",Images[I_Count]);
        imshow("Extract", M);
        waitKey(0);
    }
    return 0;
}
