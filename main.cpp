#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "usage:./cam_calibrate imgFolder_PATH" << std::endl;
    std::string imgFolder_PATH = argv[1];
    std::string imgFileName;
    std::string imgFilePath;
    std::ifstream fin(imgFolder_PATH + "imgFileNameList.txt");//open img files list
    int image_count = 0;
    int corner_count = 0;
    //int corner_size_width=argv[2];
    //int corner_size_height=argv[3];
    cv::Size corner_size(9, 6);
    cv::Size img_size;
    int corner_num = corner_size.width * corner_size.height;
    std::vector<std::vector<cv::Point2f>> img_points_seq;//img seq corner
    while (std::getline(fin, imgFileName))//progress each img
    {
        std::cout << "processing: " << imgFileName << std::endl;
        imgFilePath = imgFolder_PATH + imgFileName;
        cv::Mat img = cv::imread(imgFilePath);
        if (img.data == 0)std::cout << "[error]read img failed" << std::endl;
        img_size.width = img.cols;
        img_size.height = img.rows;
        std::cout << imgFileName << "width:" << img_size.width << std::endl;
        std::cout << imgFileName << "height:" << img_size.height << std::endl;
        std::vector<cv::Point2f> img_points_buf;//single img corner
        cv::findChessboardCorners(img, corner_size, img_points_buf);
        std::cout << img_points_buf << std::endl;
        std::cout << "----------------------------" << std::endl;
        img_points_seq.push_back(img_points_buf);
        cv::drawChessboardCorners(img, corner_size, img_points_buf, 1);
        cv::namedWindow(imgFileName, 1);//flag=0:自适应窗口大小
        cv::imshow(imgFileName, img);
        cv::waitKey(100);
        image_count++;
    }
    cv::Size square_size(32, 32);//real size(mm)
    std::vector<std::vector<cv::Point3f>> object_points;//real 3D coord(all img)
    cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 摄像机内参数矩阵 */
    //std::vector<int> point_counts;  // 每幅图像中角点的数量
    cv::Mat distCoeffs = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0)); /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3*/
    std::vector<cv::Mat> rvecsMat;  /* 每幅图像的旋转向量 */
    std::vector<cv::Mat> tvecsMat; /* 每幅图像的平移向量 */
    /* 初始化标定板上角点的三维坐标 */
    int i, j, k;
    for (k = 0; k < image_count; k++) {
        std::vector<cv::Point3f> tempPointSet;//per img -> rows ->cols
        for (i = 0; i < corner_size.height; i++) {
            for (j = 0; j < corner_size.width; j++) {
                cv::Point3f realPoint;
                /* 假设标定板放在世界坐标系中z=0的平面上 */
                realPoint.x = i * square_size.width;
                realPoint.y = j * square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        std::cout << tempPointSet << std::endl;
        std::cout << "----------------------------" << std::endl;
        object_points.push_back(tempPointSet);
    }
    cv::calibrateCamera(object_points, img_points_seq, img_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
    std::cout << "cameraMatrix:" << std::endl;
    std::cout << cameraMatrix << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "distCoeffs:" << std::endl;
    std::cout << distCoeffs << std::endl;
    std::cout << "----------------------------" << std::endl;
    for (int i = 1; i <= image_count; i++) {
        std::cout << "    " << i << std::endl;
        std::cout << "rvecsMat:" << std::endl;
        std::cout << rvecsMat[i - 1] << std::endl;
        std::cout << "" << std::endl;
        std::cout << "tvecsMat:" << std::endl;
        std::cout << tvecsMat[i - 1] << std::endl;
        std::cout << "----------------------------" << std::endl;
    }
    double error = 0;
    std::vector<cv::Point2f> virtual_2d;
    for(int i=1;i<=image_count;i++){
        std::vector<cv::Point3f> temp_object_point=object_points[i-1];
        cv::projectPoints(temp_object_point,rvecsMat[i-1],tvecsMat[i-1],cameraMatrix,distCoeffs,virtual_2d);
        std::vector<cv::Point2f> real_2d =img_points_seq[i-1];
        cv::Mat real_2d_Mat=cv::Mat(1,real_2d.size(),CV_32FC2);
        cv::Mat virtual_2d_Mat=cv::Mat(1,virtual_2d.size(),CV_32FC2);
        for(int j=0;j<real_2d.size();j++){
            virtual_2d_Mat.at<cv::Vec2f>(0,j)=cv::Vec2f(virtual_2d[j].x,virtual_2d[j].y);
            real_2d_Mat.at<cv::Vec2f>(0,j)=cv::Vec2f(real_2d[j].x,real_2d[j].y);
        }
        error=cv::norm(virtual_2d_Mat,real_2d_Mat,cv::NORM_L2);
        std::cout<<"The "<<i<<"th img error is:"<<error<<"pixels"<<std::endl;
    }
    return 0;
}