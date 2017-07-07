#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>

struct imagenData{
    std::string classname;
    cv::Mat feature;
};



namespace  mifilesystem = boost::filesystem;

typedef std::vector<std::string>::const_iterator vec_iter;
inline std::string getClassName(const std::string&nombrePhoto){
    return nombrePhoto.substr(nombrePhoto.find_last_of('/')+1,6);
}
cv::Mat getDescriptors(const cv::Mat& img){
    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    kaze->detectAndCompute(img,cv::noArray(),keypoints,descriptors);
    return descriptors;
}
std::vector<std::string> getArchivosDirectorio(const std::string & directory){
   std::vector<std::string> files;
   mifilesystem::path root(directory);
   mifilesystem::directory_iterator it_end;
       // int x=1;
  for (mifilesystem::directory_iterator it(root); it != it_end; ++it)
  {
      
      //if (boost::filesystem::is_regular_file(it->path()))
      //{
         
          files.push_back(it->path().string());
     // }
  }
       
        
  return files;
    
    
}



void leerPhotos(vec_iter begin,vec_iter end, std::function<void(const std::string&,const cv::Mat&)> callback){
 int x=0;
    for (auto it=begin;it!=end;++it){
       x++;
if (x%50==0){

std::cout<<"Hasta el momento se han leido:  "<<x<< " imagenes"<<std::endl;

}
       
        std::string nombrePhoto = *it;
        std::cout<<"Leyendo photo "<<nombrePhoto<<"..."<<std::endl;
        cv ::Mat img = cv::imread(nombrePhoto,0);
        //0 significa que la imagen sera tomada en escala de grises.
        if(img.empty()){
            std::cerr<<"Nose pudo leer imagen"<<std::endl;
            continue;
        }
        std::string nombreClass = getClassName(nombrePhoto);
        cv::Mat descriptors = getDescriptors(img);
        callback(nombreClass,descriptors);
    }
}

int getClassId(const std::set<std::string>& classes,const std::string& classname){
    int index=0;
    for(auto it = classes.begin();it !=classes.end();++it){
        if (*it==classname)break;
        ++index;
    }
    return index;
}
//regresa un binary del class
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname){
    cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(),1),CV_32F);
    int index = getClassId(classes, classname);
    code.at<float>(index)=1;
    return code;
}

cv::Ptr<cv::ml::ANN_MLP> getRedEntrenada(const cv::Mat& muestrasTrain,const cv::Mat& respuestasTrain){
    int networkInputSize = muestrasTrain.cols;
    int networkOutputSize= respuestasTrain.cols;
    cv::Ptr<cv::ml::ANN_MLP>mlp = cv::ml::ANN_MLP::create();
    std::vector<int> tamanoCapas = {networkInputSize,networkInputSize/2,networkOutputSize};
    mlp->setLayerSizes(tamanoCapas);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->train(muestrasTrain,cv::ml::ROW_SAMPLE,respuestasTrain);
    return mlp;
}


cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
                      int vocabularySize){
    cv::Mat salida= cv::Mat::zeros(cv::Size(vocabularySize,1),CV_32F);
    std::vector<cv::DMatch> matches;
    flann.match(descriptors,matches);
    for (size_t j=0;j<matches.size();j++){
        int palabraVisual=matches[j].trainIdx;
        salida.at<float>(palabraVisual)++;
    }
    return salida;
}

int getClassesPredecidas(const cv::Mat& predictions){
    float maxPrediction = predictions.at<float>(0);
    float maxPredictionIndex=0;
    const float* ptrPredictions = predictions.ptr<float>(0);
    for(int i=0;i<predictions.cols;i++){
        float prediction=*ptrPredictions++;
        if (prediction>maxPrediction){
            maxPrediction = prediction;
          maxPredictionIndex = i;
        }
    }
    return maxPredictionIndex;
}


std::vector<std::vector<int>> getMatrizConfusion(cv::Ptr<cv::ml::ANN_MLP> mlp,
                                                const cv::Mat& muestrasTest,const std::vector<int>& testSalidaEsperada){
    cv::Mat testOutput;
    mlp->predict(muestrasTest,testOutput);
   
    std::vector<std::vector<int>> confusionMatrix(2,std::vector<int>(2));
    for(int i=0;i<testOutput.rows;i++){
        int classePredicted = getClassesPredecidas(testOutput.row(i));
        
        int classeEsperada = testSalidaEsperada.at(i);
        confusionMatrix[classeEsperada][classePredicted]++;
        
    }
    return confusionMatrix;
}

void guardarModelos(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary, const std::set<std::string>& classes){
    mlp->save("mlp.yam");
    cv::FileStorage fs("vocabulary.yaml",cv::FileStorage::WRITE);
    fs<<"vocabulary"<<vocabulary;
    fs.release();
    std::ofstream classesOutput("classes.txt");
    for (auto it = classes.begin(); it != classes.end();++it){
        classesOutput << getClassId(classes,*it) <<"\t"<<*it<<std::endl;    
    }
    classesOutput.close();    
}

void imprimirMatriz(const std::vector<std::vector<int> >& confusionMatriz,
                   const std::set<std::string>& classes){
    for(auto it = classes.begin();it != classes.end();++it){
        std::cout << *it << " ";
    }
    std::cout << std::endl;
  for (size_t i = 0; i < confusionMatriz.size(); i++)
  {
      for (size_t j = 0; j < confusionMatriz[i].size(); j++)
      {
          std::cout<<"  " << confusionMatriz[i][j] << "\t";
      }
      std::cout << std::endl;
  }
}

float getExactitud(const std::vector<std::vector<int> >&confusionMatriz){
    int x=0;
    int total=0;
     for (size_t i = 0; i < confusionMatriz.size(); i++)
  {
      for (size_t j = 0; j < confusionMatriz.at(i).size(); j++)
      {
          if (i == j) x += confusionMatriz.at(i).at(j);
          total += confusionMatriz.at(i).at(j);
      }
  }
  return x / (float)total;
}


int main(int argc, char** argv){
//el programa nenecita 3 entradas por consola
    //1.-directorio de las fotos
    //2.-tamano de la capa de red
    //3.-porcentaje train/test


    if(argc!=4){
        std::cout<<"Para usar necesita: <IMAGES_DIRECTORY> <TAMANO DE LA CAPA DE RED> <PORCENTAJE DE ENTRENAMIENTO>"<<std::endl;        
        exit(-1);
    }
    // si todo se cumple
    std::string directorioImagenes = argv[1];
    int sizeNetwork = atoi(argv[2]);
    float trainSplit = atof(argv[3]);
    
    std::cout <<"Leyendo el set"<<std::endl;
    //tomo el tiempo de inicio
    double start = (double)cv::getTickCount();
    
    std::vector<std::string> files = getArchivosDirectorio(directorioImagenes);
    std::cout<<"Fotos encontradas:"<<files.size()<<std::endl;
    //se randomiza las posiciones para evitar una subida de bias  ||||porque?
    std::random_shuffle(files.begin(),files.end());
    
    ////////////////////////////////////
    
    cv::Mat descriptorsSet;
    std::vector<imagenData*> descriptorsData;
    
    std::set<std::string> misClasses; //clases que se definen en las etiquetas del nombre
    
    leerPhotos(files.begin(),files.begin()+(size_t)(files.size()*trainSplit),
              [&](const std::string& classname, const cv::Mat& descriptors){
        //obtengo mis classes (y)
        misClasses.insert(classname);
        //
        descriptorsSet.push_back(descriptors);
        
        imagenData* data = new imagenData;
        data->classname=classname;
        data->feature = cv::Mat::zeros(cv::Size(sizeNetwork,1),CV_32F);
        for (int j =0;j<descriptors.rows;j++){
            descriptorsData.push_back(data);
        }
        
    });
   std::cout <<"****************************************************************"<<std::endl;
double primeraLectura = ((double)cv::getTickCount()-start)/cv::getTickFrequency()/60;
   std::cout <<"Tiempo invertido en minutos:\t"<<primeraLectura<<std::endl;
   
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
    //PREPArando BAG OF WORDS
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////  
     std::cout <<"****************************************************************"<<std::endl;
    std::cout<<"Creando vocabulario"<<std::endl;
    start = (double)cv::getTickCount();
    cv::Mat labels;
    cv::Mat vocabulary;
    //se usa k-means para encontrar k centroides(palabras del vocabulario)
    //k-means es un algoritmo de agrupamiento
    cv::kmeans(descriptorsSet,sizeNetwork,labels,cv::TermCriteria(cv::TermCriteria::EPS+ cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
    //ya no se necesita la memoria ocupada por descriptorsSet
    descriptorsSet.release();
 std::cout <<"****************************************************************"<<std::endl;
double segundaLectura= ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;
    std::cout << "Tiempo invertido en minutos:\t" << segundaLectura<< std::endl;
    //se convierten las caracteristicas locales de cada imagen en un solo descriptor
    //aqui va la tecnica de bag of words
 std::cout <<"****************************************************************"<<std::endl;
    std::cout<<"Obteniendo histogramas..."<<std::endl;
    int* ptrLabels=(int*)(labels.data);
    int size=labels.rows*labels.cols;
    for (int i=0;i<size;i++){
        int label=*ptrLabels++;
        imagenData* data = descriptorsData[i];
        data->feature.at<float>(label)++;
    }
    //lenovo y50- 70
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
    //PREPARANDO RED NEURONAL
    //Preparando KNN
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
 std::cout <<"****************************************************************"<<std::endl;
std::cout <<"Preparando red neural..."<<std::endl;
    cv::Mat muestrasTrain;
    cv::Mat respuestasTrain;
    
    std::set<imagenData*>uniqueMetadata(descriptorsData.begin(),descriptorsData.end());
    for (auto it= uniqueMetadata.begin();it!= uniqueMetadata.end();)
    {
        imagenData* data =*it;
        cv::Mat histogramaNormalizado;
        cv::normalize(data->feature,histogramaNormalizado,0,data->feature.rows,cv::NORM_MINMAX,-1,cv::Mat());
        //dato mUestras Train
        muestrasTrain.push_back(histogramaNormalizado);
        //
        respuestasTrain.push_back(getClassCode(misClasses,data->classname));
        delete *it;
        it++;
        
    }
    descriptorsData.clear();
    
    
 /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
    //ENTRENANDO RED NEURONAL o knn
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
 std::cout <<"****************************************************************"<<std::endl;
     std::cout<<"Entrenando red neural..."<< std::endl;
    start = cv::getTickCount();
    cv::Ptr<cv::ml::ANN_MLP>mlp= getRedEntrenada(muestrasTrain,respuestasTrain);

double terceraLectura =((double)cv::getTickCount()-start)/cv::getTickFrequency()/60;
    std::cout<<"Timepo invertido en minutos:\t"<<terceraLectura<<std::endl;
    //limpiarMemoria
    muestrasTrain.release();
    respuestasTrain.release();
    
    ///////////////////////////////////////////////
    /////////////////////////////////////////////
     //MODELO FLANN
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
 std::cout <<"****************************************************************"<<std::endl;
       std::cout <<"Entrenando FLANN..."<<std::endl;
    start = cv::getTickCount();
    cv::FlannBasedMatcher flann;
    flann.add(vocabulary);
    flann.train();
 std::cout <<"****************************************************************"<<std::endl;
double cuartaLectura = ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0;
    std::cout<<"Tiempo invertido en minutos:..."<< cuartaLectura<< std::endl;

    ///////////////////////////////////////////////
    /////////////////////////////////////////////
     //MODELO FLANN
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
 std::cout <<"****************************************************************"<<std::endl;
    std::cout<<"Leiendo test..." <<std::endl;
    start = cv::getTickCount();
    cv::Mat muestrasTest;
    std::vector<int> testSalida;
   
    leerPhotos(files.begin() + (size_t)(files.size() * trainSplit),files.end(),
               [&](const std::string& classname,const cv::Mat&descriptors){
        cv::Mat bowFeatures= getBOWFeatures(flann,descriptors,sizeNetwork);
        cv::normalize(bowFeatures,bowFeatures,0,bowFeatures.rows,cv::NORM_MINMAX,-1,cv::Mat());
        muestrasTest.push_back(bowFeatures);
        testSalida.push_back(getClassId(misClasses,classname));
    });
 std::cout <<"****************************************************************"<<std::endl;
double quintaLectura=((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 ;
          std::cout << "Tiempo invertido en minutos:\t" << quintaLectura<< std::endl;
        

    ///////////////////////////////////////////////
    /////////////////////////////////////////////
     //MATRIZ DE CONFUSION
    /////////////////////////////////////////////////////
    ////////////////////////////////////////////////////
    
    std::vector<std::vector<int> >confusionMatriz = getMatrizConfusion(mlp,muestrasTest,testSalida);
    
    //OBTENER EXACTITUD
 std::cout <<"****************************************************************"<<std::endl;
    std::cout<<"Matriz de Confusion: "<<std::endl;
    imprimirMatriz(confusionMatriz,misClasses);
    std::cout<<"Exactitud: "<<getExactitud(confusionMatriz)<<std::endl;
    
    //////////////////////////////////////////////
    //////////////////////////////////////////////
    //GUARDANDO MODELO
    /////////////////////////////////////////////
    /////////////////////////////////////////////
    
    
    std::cout<<"Guardando Modelos"<<std::endl;
    guardarModelos(mlp,vocabulary,misClasses);
 std::cout <<"*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"<<std::endl;
 std::cout <<"------------------------FIN DEL PROCESO--------------------------"<<std::endl;
 std::cout <<"*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*"<<std::endl;
 std::cout <<"Tiempo invertido para leer las imagenes para el entrenamiento"<<std::endl;
 std::cout <<primeraLectura<<std::endl;
 std::cout <<"Tiempo invertido para la creaion del vocabulario"<<std::endl;
 std::cout <<segundaLectura<<std::endl;
 std::cout <<"Tiempo invertido para el TRAIN de la red neuronal MLP"<<std::endl;
 std::cout <<terceraLectura<<std::endl;
 std::cout <<"Tiempo invertido para el entrenamiento de FLANN"<<std::endl;
 std::cout <<cuartaLectura<<std::endl;
 std::cout <<"Tiempo invertido para el TEST"<<std::endl;
 std::cout <<quintaLectura<<std::endl;
    
   return 0;
    
}

               

//consultar cv::normilaze













