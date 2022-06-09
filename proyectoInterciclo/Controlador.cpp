#include "Modelo.hpp"

int thres = 1;
int thresC1 = 1;

Mat image0;


VideoCapture real_time("/dev/video0");
Mat video_stream;
Mat corte;
Mat imagen;
Mat color;
Mat color2;
Mat suma;
Mat suma2;
Mat r1;
Mat r2;
Mat elemento ;
Mat salida;
//----------------
double alpha;
double bet;
Mat dst;
//------------
Mat frameMediana;
Mat salidaF;
Mat salidaF2;
Mat salidaI;
Mat salidaI2;
Mat salidaI3;
Mat salidaI4;
Mat salidaI5;
Mat salidaI6;
Mat salidaI7;
Mat salidaI8;
Mat salidaI9;
Mat salidaI10;

void eventoTrack(int v, void*data){ 
    
}
void eventoTrackColor1(int v, void*data){ 
}


Mat detectar(Mat img){
string trained_classifier_location = "/home/computacion/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
CascadeClassifier faceDetector;
faceDetector.load(trained_classifier_location);
vector<Rect>cara;
    while (true) {
        faceDetector.detectMultiScale(img, cara, 1.1, 6, CASCADE_SCALE_IMAGE, Size(250, 250));
        real_time.read(img);
        corte=img.clone();
        for (int i = 0; i < cara.size(); i++){
            Mat faceROI = img(cara[i]);
            int x = cara[i].x;
            int y = cara[i].y;
            int h = y + cara[i].height;
            int w = x + cara[i].width;
            Rect co(Point(x,y),Point(w,h));
            rectangle(img, co, Scalar(255, 0, 255), 2, 8, 0);
            if(co.width>0 && co.height>0){
            }
            color=imread("ColorRojo.jpg");
            resize(color,color,Size(corte(co).rows,corte(co).cols));
            elemento = getStructuringElement(MORPH_CROSS, Size(50, 50));
            
            add(corte(co),color,suma);
            morphologyEx(suma,suma, MORPH_TOPHAT, elemento, Point(-1,-1), 1);
            cvtColor(suma,suma,COLOR_BGR2HSV);
            
            return suma;
        }
    }
}

//---------------------------------------
Mat capturar(Mat img){
    Mat entrada = detectar(img);
    color=imread("ColorRojo.jpg");
    resize(color,color,Size(entrada.rows,entrada.cols));
    add(entrada,color,r1);
    return r1;

}
    
Mat capturar2(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorAzul.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,r2);
    return r2;
}

Mat capturar3(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorNegro.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

Mat capturar4(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorVerde.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

Mat capturar5(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorAmarillo.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

Mat capturar6(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorMorado.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

Mat capturar7(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorNaranja.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

Mat capturar8(Mat img){
    Mat entrada2 = detectar(img);
    color=imread("ColorCafe.jpg");
    resize(color,color,Size(entrada2.rows,entrada2.cols));
    add(entrada2,color,suma2);
    cvtColor(suma2,suma2,COLOR_BGR2HSV);
    return suma2;
}

//--------------------------------------------------------------------------------------------------------------------
Mat Intensidad(Mat imagen){


    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar(entrada2);
    
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    

    
    return salida;


}

Mat Intensidad2(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar2(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);

    return salida;
}

Mat Intensidad3(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar3(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}

Mat Intensidad4(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar4(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}
Mat Intensidad5(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar5(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}

Mat Intensidad6(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar6(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}

Mat Intensidad7(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar7(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}

Mat Intensidad8(Mat imagen){
    Mat entrada2 = detectar(imagen);
    Mat entradaI = capturar8(entrada2);
    Mat salida;
    alpha = (double) thresC1/55;
    bet = ( 1.0 - alpha );
    addWeighted(entradaI, alpha, color, bet, 0.0, salida);
    return salida;
}



void getSquareImage(cv::InputArray img, cv::OutputArray dst, int size)
{
    if (size < 2) size = 2;
    int width = img.cols(), height = img.rows();

    cv::Mat square = dst.getMat();

    // si la imagen es cuadrada solo redimensionar
    if (width == height) {
        cv::resize(img, square, Size(size, size));
        return;
    }

    // establecer color de fondo del cuadrante
    square.setTo(Scalar::all(0));

    int max_dim = (width >= height) ? width : height;
    float scale = ((float)size) / max_dim;

    // calcular la region centrada 
    cv::Rect roi;

    if (width >= height)
    {
        roi.width = size;
        roi.x = 0;
        roi.height = (int)(height * scale);
        roi.y = (size - roi.height) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = size;
        roi.width = (int)(width * scale);
        roi.x = (size - roi.width) / 2;
    }

    // redimensionar imagen en la region calculada
    cv::resize(img, square(roi), roi.size());
}

void showImages(const String& window_name, int rows, int cols, int size, std::initializer_list<const Mat*> images, int pad = 1)
{
    if (pad <= 0) pad = 0;

    int width = size * cols + ((cols + 1) * pad);
    int height = size * rows + ((rows + 1) * pad);

    // crear la imagen de salida con un color de fondo blanco
    Mat dst = Mat(height, width, CV_8UC3, Scalar::all(0));

    int x = 0, y = 0, cols_counter = 0, img_counter = 0;

    // recorrer la lista de imagenes
    for (auto& img : images) {
        Mat roi = dst(Rect(x + pad, y + pad, size, size));
        
        // dibujar la imagen en el cuadrante indicado
        getSquareImage(*img, roi, size);

        // avanzar al siguiente cuadrante
        x += roi.cols + pad;

        // avanza a la siguiente fila
        if (++cols_counter == cols) {
            cols_counter = x = 0;
            y += roi.rows + pad;
        }

        // detener si no hay mas cuadrantes disponibles
        if (++img_counter >= rows * cols) break;
    }
    imshow(window_name, dst);
}

Mat reduccionLuz(Mat imag0){

            Mat frameMediana;
            Mat frameMediana2;
            Mat frameMediana3;
            Mat frameMediana4;
            Mat frameMediana5;
            Mat frameMediana6;
            Mat frameMediana7;
            Mat frameMediana8;
            Mat frameMediana9;
            Mat frameMediana10;

            medianBlur(imag0,frameMediana,5);
            medianBlur(frameMediana,frameMediana2,5);
            medianBlur(frameMediana2,frameMediana3,5);                    
            medianBlur(frameMediana3,frameMediana4,5);                    
            medianBlur(frameMediana4,frameMediana5,5);                    
            medianBlur(frameMediana5,frameMediana6,5);                    
            medianBlur(frameMediana6,frameMediana7,5);                    
            medianBlur(frameMediana7,frameMediana8,5);                    
            medianBlur(frameMediana8,frameMediana9,5);                    

return frameMediana9;
}


void Modelo::transformar(){

    if(real_time.isOpened()){
      int frame_width = real_time.get(cv::CAP_PROP_FRAME_WIDTH);

	  int frame_height = real_time.get(cv::CAP_PROP_FRAME_HEIGHT);

	   
        namedWindow("Mosaico", WINDOW_AUTOSIZE);
        namedWindow("Original", WINDOW_AUTOSIZE);
      	VideoWriter video("videoOriginal.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(frame_width,frame_height));
      	VideoWriter video1("videoEfectoRojo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
        VideoWriter video2("videoEfectoAzul.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
      	VideoWriter video3("videoEfectoNegro.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
        VideoWriter video4("videoEfectoVerde.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
      	VideoWriter video5("videoEfectoAmarillo.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
        VideoWriter video6("videoEfectoMorado.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
      	VideoWriter video7("videoEfectoNaranja.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));
        VideoWriter video8("videoEfectoCafe.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, Size(320,320));

        Mat videoJ;
        Mat videoJ2;

        while(3==3){

            real_time >> image0;

            showImages("Original", 1, 1, 500, { &image0 }, 5);
            createTrackbar("Opciones", "Original", &thres, 5, eventoTrack, NULL);
            createTrackbar("Intesidad","Original",&thresC1,150,eventoTrackColor1,NULL);
            video.write(image0);
            

            if(waitKey(33)==27){
                break;
            }

                                    


            switch(thres){
                case 1:
                    showImages("Mosaico", 1, 1, 480, { &image0}, 5);
                    break;
                case 2:

                    frameMediana = reduccionLuz(image0);
                    salidaI= Intensidad(frameMediana);
                    salidaI2= Intensidad2(frameMediana);
                    resize(image0,image0,Size(320,320));
                    resize(salidaI,salidaI,Size(320,320));
                    resize(salidaI2,salidaI2,Size(320,320));  

                                    
                    video1.write(salidaI);
                    video2.write(salidaI2);
                    hconcat(image0,salidaI,videoJ);
                    resize(videoJ,videoJ,Size(320,320));  
                           
                    showImages("Mosaico", 1, 2, 240, { &salidaI, &salidaI2}, 5);

                                        
                    break;

                case 3:
                    frameMediana = reduccionLuz(image0);
                    salidaI= Intensidad(frameMediana);
                    salidaI2= Intensidad2(frameMediana);
                    salidaI3= Intensidad3(frameMediana);
                    salidaI4= Intensidad4(frameMediana);

                    resize(salidaI,salidaI,Size(320,320));
                    resize(salidaI2,salidaI2,Size(320,320));                    
                    resize(salidaI3,salidaI3,Size(320,320));
                    resize(salidaI4,salidaI4,Size(320,320));                    

                    video1.write(salidaI);
                    video2.write(salidaI2);  
                    video3.write(salidaI3);
                    video4.write(salidaI4);  

                    showImages("Mosaico", 2, 2, 240, { &salidaI, &salidaI2, &salidaI3, &salidaI4}, 5);
                    break;
                case 4:
                    frameMediana = reduccionLuz(image0);
                    salidaI= Intensidad(frameMediana);
                    salidaI2= Intensidad2(frameMediana);
                    salidaI3= Intensidad3(frameMediana);
                    salidaI4= Intensidad4(frameMediana);
                    salidaI5= Intensidad5(frameMediana);
                    salidaI6= Intensidad6(frameMediana);

                    resize(salidaI,salidaI,Size(320,320));
                    resize(salidaI2,salidaI2,Size(320,320));                    
                    resize(salidaI3,salidaI3,Size(320,320));
                    resize(salidaI4,salidaI4,Size(320,320));                    
                    resize(salidaI5,salidaI5,Size(320,320));
                    resize(salidaI6,salidaI6,Size(320,320));                    

                    video1.write(salidaI);
                    video2.write(salidaI2);  
                    video3.write(salidaI3);
                    video4.write(salidaI4);  
                    video5.write(salidaI5);
                    video6.write(salidaI6);  

                    showImages("Mosaico", 2, 3, 240, { &salidaI, &salidaI2, &salidaI3, &salidaI4, &salidaI5, &salidaI6 }, 5);
                    break;
                case 5:
                    frameMediana = reduccionLuz(image0);
                    salidaI= Intensidad(frameMediana);
                    salidaI2= Intensidad2(frameMediana);
                    salidaI3= Intensidad3(frameMediana);
                    salidaI4= Intensidad4(frameMediana);
                    salidaI5= Intensidad5(frameMediana);
                    salidaI6= Intensidad6(frameMediana);
                    salidaI7= Intensidad7(frameMediana);
                    salidaI8= Intensidad8(frameMediana);


                    resize(salidaI,salidaI,Size(320,320));
                    resize(salidaI2,salidaI2,Size(320,320));                    
                    resize(salidaI3,salidaI3,Size(320,320));
                    resize(salidaI4,salidaI4,Size(320,320));                    
                    resize(salidaI5,salidaI5,Size(320,320));
                    resize(salidaI6,salidaI6,Size(320,320));                    
                    resize(salidaI7,salidaI7,Size(320,320));
                    resize(salidaI8,salidaI8,Size(320,320));                    

                    video1.write(salidaI);
                    video2.write(salidaI2);  
                    video3.write(salidaI3);
                    video4.write(salidaI4);  
                    video5.write(salidaI5);
                    video6.write(salidaI6);
                    video7.write(salidaI7);
                    video8.write(salidaI8);  
                    showImages("Mosaico", 2, 4, 240, { &salidaI, &salidaI2, &salidaI3, &salidaI4, &salidaI5, &salidaI6, &salidaI7, &salidaI8}, 5);
                    break;
                    
            }
        }


        real_time.release();
        video.release();
        video1.release();
        video2.release();
        video3.release();
        video4.release();
        video5.release();
        video6.release();
        video7.release();
        video8.release();
 

        destroyAllWindows();

    }    
}


