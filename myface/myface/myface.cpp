// myface.cpp: ����Ŀ�ļ���

#include "stdafx.h"

#include <cstring>        // for strcat()
#include <io.h>

#include<opencv2\opencv.hpp>  
#include<opencv2\face.hpp>
#include<iostream>  

using namespace std;
using namespace cv;
using namespace cv::face;


//string opencv_face_xml = "lbpcascade_frontalface.xml";
string opencv_face_xml = "haarcascade_frontalface_alt.xml";

VideoCapture cap(0);    //��Ĭ������ͷ  

struct face_st
{
	String name;
};

struct face_st face_array[1024];

void face_array_init(void)
{
	int i;
	for (i = 0; i < sizeof(face_array) / sizeof(face_array[0]); i++)
	{
		if (i == 41)
		{
			face_array[i].name = "lianzhian";
		}
		else
		{
			face_array[i].name = "undef";
		}
		
	}
}

int take_image(int rflg, int index);
int train_face(void);
String get_face_name(int index);
void get_input_name(int index);

int main()
{
	if (!cap.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat edges;
	Mat gray;
	
	int perv_predictPCA = 0;
	int count = 0;

	CascadeClassifier cascade;
	bool stop = false;

	face_array_init();

	//ѵ���õ��ļ����ƣ������ڿ�ִ���ļ�ͬĿ¼��  
	cascade.load(opencv_face_xml);
	//cascade.load("haarcascade_frontalface_alt");

	Ptr<FaceRecognizer> modelPCA = EigenFaceRecognizer::create();
	//modelPCA->load("MyFacePCAModel.xml");
	modelPCA->read("MyFaceFisherModel.xml");

	while (1)
	{
		cap >> frame;

		//�������ڴ����������������  
		vector<Rect> faces(0);

		cvtColor(frame, gray, CV_BGR2GRAY);
		//�ı�ͼ���С��ʹ��˫���Բ�ֵ  
		//resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);  
		//�任���ͼ�����ֱ��ͼ��ֵ������  
		equalizeHist(gray, gray);

		cascade.detectMultiScale(gray, faces,
			1.1, 2, 0
			//|CV_HAAR_FIND_BIGGEST_OBJECT  
			//|CV_HAAR_DO_ROUGH_SEARCH  
			| CV_HAAR_SCALE_IMAGE,
			Size(30, 30));

		Mat face;
		Point text_lb;

		for (size_t i = 0; i < faces.size(); i++)
		{
			if (faces[i].height > 0 && faces[i].width > 0)
			{
				face = gray(faces[i]);
				text_lb = Point(faces[i].x, faces[i].y);

				rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);

				Mat face_test;

				int predictPCA = 0;
				if (face.rows >= 120)
				{
					resize(face, face_test, Size(92, 112));

				}
				//Mat face_test_gray;  
				//cvtColor(face_test, face_test_gray, CV_BGR2GRAY);  

				if (!face_test.empty())
				{
					//����ͼ��Ӧ���ǻҶ�ͼ  
					predictPCA = modelPCA->predict(face_test);
				}

				cout << predictPCA << endl;

				if (predictPCA != 0)
				{
					if (predictPCA == perv_predictPCA)
					{
						count++;

						/*
						string name = "lianzhian";
						putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
						*/
					}
					else
					{
						count = 0;
						perv_predictPCA = predictPCA;
					}

					//if (count >= 3)
					{
						//string name = face_array[predictPCA].name;
						string name = get_face_name(predictPCA);
						/*
						switch (predictPCA)
						{
							case 41:
								name = "lianzhian";
								break;
							default:
								break;
						}
						*/
				
						putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
					}
				}
			}
		}
		imshow("face", frame);
		
		switch (char(waitKey(200)))
		{
			case 'p':
			{
				int index;
				index = take_image(0, 0);

				if (index != -1)
				{
					train_face();
					get_input_name(index);

					modelPCA->read("MyFaceFisherModel.xml");
				}
			}
			break;

			case 't':
			{
				train_face();

				modelPCA->read("MyFaceFisherModel.xml");
			}
			break;

			case 'r':
			{
				take_image(1, 41);
			}

			break;
		}


	}

	return 0;
}


int check_dir_count(const char * dir)
{
	int dir_count = 0;
	char dirNew[200];
	strcpy(dirNew, dir);
	strcat(dirNew, "\\*.*");    // ��Ŀ¼�������"\\*.*"���е�һ������

	intptr_t handle;
	_finddata_t findData;

	handle = _findfirst(dirNew, &findData);
	if (handle == -1)        // ����Ƿ�ɹ�
		return -1;

	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
				continue;

			dir_count++;
			/*
			cout << findData.name << "\t<dir>\n";

			// ��Ŀ¼�������"\\"����������Ŀ¼��������һ������
			strcpy(dirNew, dir);
			strcat(dirNew, "\\");
			strcat(dirNew, findData.name);

			listFiles(dirNew);
			*/

		}
		/*
		else
			
			cout << findData.name << "\t" << findData.size << " bytes.\n";
			*/
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);    // �ر��������

	return dir_count;
}

/*
rflg : ��ʾҪ��Ҫ���²ɼ����ݣ����Ϊ 1 ��ʾ���²ɼ�
index ����ʾ���²ɼ�˭��
*/
int take_image(int rflg, int index)
{
	CascadeClassifier cascade;
	cascade.load(opencv_face_xml);
	
	
	Mat frame;
	int pic_num = 1;
	int dir_count;

	if (rflg == 1)
	{
		dir_count = index;
	}else
	{
		dir_count = check_dir_count("att_faces") + 1;

		if (dir_count == -1)
			return -1;


		cout << "dir_count is :" << dir_count << endl;

		string dir_name = format("att_faces\\s%d", dir_count);

		cout << "mkdir dir :" << dir_name << endl;

		string command;
		command = "mkdir " + dir_name;
		system(command.c_str());
	}


	while (1)
	{
		cap >> frame;

		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

		cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0, Size(100, 100), Size(500, 500));

		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
		}

		if (faces.size() == 1)
		{
			Mat faceROI = frame_gray(faces[0]);
			Mat myFace;
			resize(faceROI, myFace, Size(92, 112));
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, (0, 0, 255), 2, LINE_AA);

			string filename = format("att_faces\\s%d\\new%d.jpg", dir_count, pic_num);
			cout << "filename is :" << filename << endl;
			imwrite(filename, myFace);

			//��������²ɼ����Ͳ���Ҫ�����ˡ�
			if (rflg == 0)
			{
				ofstream write("at.txt", ios::app);//��record.txt�ļ�����ios::app׷�ӵķ�ʽ����
				write << filename << ";" << dir_count << endl;
				write.close();//�ر��ļ�
			}
			imshow(filename, myFace);
			waitKey(500);
			destroyWindow(filename);
			
			pic_num++;
			if (pic_num == 11)
			{
				destroyWindow("frame");
				return dir_count;
			}
		}
		imshow("frame", frame);
		waitKey(100);
	}
	return dir_count;
}




static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// �����ͷ���һ����һ�����ͼ�����:  
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

//ʹ��CSV�ļ�ȥ��ͼ��ͱ�ǩ����Ҫʹ��stringstream��getline����  
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


int train_face(void)
{

	//��ȡ���CSV�ļ�·��.  
	//string fn_csv = string(argv[1]);  
	string fn_csv = "at.txt";

	// 2�����������ͼ�����ݺͶ�Ӧ�ı�ǩ  
	vector<Mat> images;
	vector<int> labels;
	// ��ȡ����. ����ļ����Ϸ��ͻ����  
	// ������ļ����Ѿ�����.  
	try
	{
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e)
	{
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// �ļ������⣬����ɶҲ�������ˣ��˳���  
		exit(1);
	}
	// ���û�ж�ȡ���㹻ͼƬ��Ҳ�˳�.  
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	// ����ļ��д�������Ǵ�������ݼ����Ƴ����һ��ͼƬ  
	//[gm:��Ȼ������Ҫ�����Լ�����Ҫ�޸ģ���������˺ܶ�����]  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// ���漸�д�����һ��������ģ����������ʶ��  
	// ͨ��CSV�ļ���ȡ��ͼ��ͱ�ǩѵ������  
	// T������һ��������PCA�任  
	//�����ֻ�뱣��10�����ɷ֣�ʹ�����´���  
	//      cv::createEigenFaceRecognizer(10);  
	//  
	// ����㻹ϣ��ʹ�����Ŷ���ֵ����ʼ����ʹ��������䣺  
	//      cv::createEigenFaceRecognizer(10, 123.0);  
	//  
	// �����ʹ��������������ʹ��һ����ֵ��ʹ��������䣺  
	//      cv::createEigenFaceRecognizer(0, 123.0);  

	cout << "train MyFacePCAModel ... pls wait " << endl;

	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	cout << "train MyFaceFisherModel ... pls wait " << endl;

	Ptr<BasicFaceRecognizer> model1 = FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	cout << "train MyFaceLBPHModel ... pls wait " << endl;

	Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
	model2->train(images, labels);
	model2->save("MyFaceLBPHModel.xml");

	// ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���  
	int predictedLabel = model->predict(testSample);
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	// ����һ�ֵ��÷�ʽ�����Ի�ȡ���ͬʱ�õ���ֵ:  
	//      int predictedLabel = -1;  
	//      double confidence = 0.0;  
	//      model->predict(testSample, predictedLabel, confidence);  

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);
	string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);
	cout << result_message << endl;
	cout << result_message1 << endl;
	cout << result_message2 << endl;

	//getchar();
	//waitKey(0);
	return 0;
}


String get_face_name(int index)
{
	string ret_string = "undef";

	string filename = "face_name.txt";
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, face_index, face_name;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, face_index, ';');
		getline(liness, face_name);

		if (atoi(face_index.c_str()) == index)
		{
			return face_name;
		}
	}

	return ret_string;
}

void get_input_name(int index)
{
	char name[512];

	cout << "Enter user name:\n";
	cin >> name;

	ofstream write("face_name.txt", ios::app);//��record.txt�ļ�����ios::app׷�ӵķ�ʽ����
	write << index << ";" << name << endl;
	write.close();//�ر��ļ�
}




