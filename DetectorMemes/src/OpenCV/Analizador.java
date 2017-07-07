/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package OpenCV;


import analizadordemomos.FrameMenu;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 *
 * @author jairo
 */
public class Analizador {
    String path ;
    
    
   
    public void getPrediccion(String path,FrameMenu menu){
        try {
           Runtime rt = Runtime.getRuntime();
           String [] comandos ={"src/AnalizadorC/aplicativo",
                                "src/ArtificialData/mlp.yam",
                                "src/ArtificialData/vocabulary.yaml",
                                "src/ArtificialData/classes.txt",
                                path};
           
           
           System.out.println(Arrays.toString(comandos));
           /*./aplicativo 
           /home/jairo/Desktop/OpenCVFinal/Flies/mlp.yam 
           /home/jairo/Desktop/OpenCVFinal/Flies/vocabulary.yaml 
           /home/jairo/Desktop/OpenCVFinal/Flies/classes.txt 
           /home/jairo/Desktop/OpenCVFinal/meme.jpg */
           Process proc = rt.exec(comandos);
           BufferedReader stdInput = new BufferedReader(new 
     InputStreamReader(proc.getInputStream()));

BufferedReader stdError = new BufferedReader(new 
     InputStreamReader(proc.getErrorStream()));
String s = "";
if ((s = stdInput.readLine()) != null) {
  menu.setResultado(s);
  
}

      
            
        } catch (IOException ex) {
            Logger.getLogger(Analizador.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        menu.setResultado("error");}
       
    }
    
    


    










