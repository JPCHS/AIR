package io.github.et.generator;

import io.github.et.generator.conponent.EventButton;
import io.github.et.tools.Client;
import javafx.animation.FadeTransition;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.ImageCursor;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.effect.InnerShadow;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
import javafx.scene.text.TextFlow;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Duration;
import kotlinx.coroutines.GlobalScope;
import kotlinx.coroutines.Job;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Objects;
import java.util.Random;

public class MainApplication extends Application {
    public static File imgFile=null;
    public static int isAI=0;
    public static Random random = new Random();
    static String a = null;
    @Override
    public void start(Stage stage) throws IOException {
        stage.setOnCloseRequest(e->{
            try {
                Client.getReply("Exit");
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
        stage.setResizable(false);
        Runtime.getRuntime().exec("./pyexec.exe");
        Image background =new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background1.png")));
        AnchorPane root1 = new AnchorPane();
        ImageView imageView=new ImageView();
        imageView.setFitWidth(720);
        imageView.setFitHeight(405);
        imageView.setImage(background);
        ImageView characterBg=new ImageView(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/character-bg.jpg"))));
        AnchorPane.setTopAnchor(characterBg,60d);
        AnchorPane.setLeftAnchor(characterBg,150d);
        characterBg.setFitWidth(450d);
        characterBg.setFitHeight(225d);
        Label num=new Label("");

        InnerShadow innerShadow = new InnerShadow();
        innerShadow.setRadius(6);
        innerShadow.setOffsetX(-2);
        innerShadow.setOffsetY(-2);
        innerShadow.setColor(Color.BLACK);
        Text text = new Text("+ 上传文件");
        text.setFont(Font.loadFont(new FileInputStream("./zh-cn.ttf"),38));
        TextFlow label = new TextFlow(text);
        AnchorPane.setLeftAnchor(label,150d);
        AnchorPane.setTopAnchor(label,150d);
        AnchorPane.setRightAnchor(label,150d);
        label.setTextAlignment(TextAlignment.CENTER);
        text.setFill(Color.rgb(70,130,180));
        text.setEffect(innerShadow);
        EventButton examine=new EventButton("检测");
        AnchorPane.setBottomAnchor(examine,50d);
        AnchorPane.setRightAnchor(examine,30d);
        FadeTransition waiting=new FadeTransition(new Duration(6050d),num);
        num.setFont(Font.loadFont(new FileInputStream("./zh-cn.ttf"),100));
        num.setTextFill(Color.BLACK);
        AnchorPane.setBottomAnchor(num,0.0);
        AnchorPane.setLeftAnchor(num,0.0);
        AnchorPane.setRightAnchor(num,0.0);
        AnchorPane.setTopAnchor(num,0.0);
        num.setAlignment(Pos.CENTER);
        num.setVisible(false);

        Label launch=new Label("A  I  R");
        launch.setFont(Font.loadFont(new FileInputStream("./zh-cn.ttf"),100));
        launch.setAlignment(Pos.CENTER);
        Label announcement=new Label("本软件所用图片、字体等素材原作者为米哈游，取自B站@难忘的旋律Official");
        announcement.setFont(Font.loadFont(new FileInputStream("./zh-cn.ttf"),15));
        announcement.setAlignment(Pos.BOTTOM_CENTER);
        FileChooser fc=new FileChooser();
        fc.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image File","*.png","*.jpg","*.gif","*.jpeg","*.bmp","*.ico","*.icon")
        );
        label.setOnMouseClicked(evt->{
            try {
                imgFile = fc.showOpenDialog(stage);
                a=imgFile.getName();
                if (a.length() > 9) {
                    a=a.substring(0, 9) + "...";
                }
                text.setFill(Color.rgb(70,130,180));
                isAI = Client.getReply(imgFile.getAbsolutePath());
                text.setText(a);
                        
                 
                text.setText(a);

            }catch(Exception ignored){}
        });


        FadeTransition waitingTask=new FadeTransition(new Duration(6050d),imageView);
        FadeTransition waitingAgain=new FadeTransition(new Duration(1000d),label);
        Button skip=new Button("跳过");
        AnchorPane.setRightAnchor(skip,0d);
        skip.setFont(Font.loadFont(new FileInputStream("./zh-cn.ttf"),15));
        skip.setOnAction(act->{
            waitingTask.stop();
            waitingAgain.stop();
            imageView.setImage(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background.png"))));
            if(isAI==0){
                num.setText("大概率是AI");
            }else if(isAI==1){
                num.setText("疑似AI");
            }else{
                num.setText("应该不是AI");
            }
            skip.setVisible(false);
            num.setVisible(true);
            num.setTextFill(Color.BLACK);
            waitingAgain.setOnFinished(m->{
                if(isAI==2){
                    num.setTextFill(Color.WHITE);
                }else if(isAI==1){
                    num.setTextFill(Color.rgb(216,191,216));
                }else{
                    num.setTextFill(Color.rgb(238,232,170));
                }
                root1.setOnMouseClicked(eb->{
                    text.setText("+ 上传文件");
                    text.setFill(Color.rgb(70,130,180));
                    imageView.setImage(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background1.png"))));
                    imgFile=null;
                    label.setVisible(true);
                    examine.setVisible(true);
                    num.setVisible(false);
                    characterBg.setVisible(true);
                    root1.setOnMouseClicked(mw->{});
                });
            });
            waitingAgain.play();
        });
        skip.setVisible(false);

        root1.getChildren().add(launch);
        root1.getChildren().add(announcement);
        AnchorPane.setLeftAnchor(launch,0.0);
        AnchorPane.setRightAnchor(launch,0.0);
        AnchorPane.setBottomAnchor(launch,0.0);
        AnchorPane.setTopAnchor(launch,0.0);
        AnchorPane.setLeftAnchor(announcement,0d);
        AnchorPane.setRightAnchor(announcement,0d);
        AnchorPane.setTopAnchor(announcement,0d);
        AnchorPane.setBottomAnchor(announcement,0d);
        root1.getChildren().add(imageView);
        root1.getChildren().add(characterBg);
        root1.getChildren().add(label);
        root1.getChildren().add(examine);
        root1.getChildren().add(num);
        root1.getChildren().add(skip);

        imageView.setVisible(false);
        characterBg.setVisible(false);
        label.setVisible(false);
        examine.setVisible(false);
        FadeTransition ft=new FadeTransition(new Duration(3000d),launch);
        FadeTransition anoFt=new FadeTransition(new Duration(3000d),announcement);
        ft.setFromValue(1.0);
        ft.setToValue(0.0);
        anoFt.setFromValue(1.0);
        anoFt.setToValue(0.0);
        ft.setOnFinished(event->{
            characterBg.setVisible(true);
            announcement.setVisible(false);
            launch.setVisible(false);
            imageView.setVisible(true);
            label.setVisible(true);
            examine.setVisible(true);
        });
        FadeTransition ft1=new FadeTransition(new Duration(3000d),launch);
        FadeTransition anoFt1=new FadeTransition(new Duration(3000d),announcement);
        ft1.setFromValue(0.0);
        ft1.setToValue(1.0);
        anoFt1.setFromValue(0.0);
        anoFt1.setToValue(1.0);
        ft1.setOnFinished(e->{
            anoFt.play();
            ft.play();
        });
        ft1.play();
        anoFt1.play();

        examine.setOnAction(event ->{
            if (imgFile!=null) {
                skip.setVisible(true);
                examine.setVisible(false);
                label.setVisible(false);
                characterBg.setVisible(false);
                Image img;
                if (isAI ==2) {
                    img = new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/1_w.gif")));
                } else if (isAI == 1) {
                    img = new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/1_p.gif")));
                } else {
                    img = new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/1.gif")));
                }
                imageView.setImage(img);
                waitingTask.setFromValue(1d);
                waitingTask.setToValue(1d);
                waitingTask.setOnFinished(e -> {
                    skip.setVisible(false);
                    imageView.setImage(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background.png"))));
                    num.setTextFill(Color.BLACK);
                    num.setVisible(true);
                    imageView.setImage(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background.png"))));
                    if (isAI ==0) {
                        num.setText("大概率是AI");
                    } else if (isAI == 1) {
                        num.setText("疑似AI");
                    } else {
                        num.setText("应该不是AI");
                    }
                    num.setVisible(true);
                    num.setTextFill(Color.BLACK);
                    waitingAgain.setOnFinished(evt -> {


                        if (isAI == 2) {
                            num.setTextFill(Color.WHITE);
                        } else if (isAI == 1) {
                            num.setTextFill(Color.rgb(216, 191, 216));
                        } else {
                            num.setTextFill(Color.rgb(238, 232, 170));
                        }
                        root1.setOnMouseClicked(eb->{
                            text.setText("+ 上传文件");
                            text.setFill(Color.rgb(70,130,180));
                            imageView.setImage(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/background1.png"))));
                            imgFile=null;
                            label.setVisible(true);
                            examine.setVisible(true);
                            num.setVisible(false);
                            characterBg.setVisible(true);
                            root1.setOnMouseClicked(mw->{});
                        });


                    });
                    waitingAgain.play();



                });
                waitingTask.play();
            }else{
                text.setFill(Color.RED);
            }

        });



        Scene scene = new Scene(root1,718, 403);
        scene.setCursor(new ImageCursor(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/cursor.png")))));
        stage.getIcons().add(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/icons/32.png"))));
        stage.getIcons().add(new Image(Objects.requireNonNull(this.getClass().getResourceAsStream("/io/et/github/generator/icons/64.png"))));
        stage.setTitle("AIR");
        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) throws IOException {
        File logTempDir=new File(System.getProperty ("user.home")+"/.generator/tempInfo");
        File logTemp=new File(System.getProperty ("user.home")+"/.generator/tempInfo/lastInfo.log");
        try{
            if(!(logTemp.exists()&&logTempDir.exists())){
                logTempDir.mkdirs();
                logTemp.createNewFile();
            }
        }catch(Exception e){
            throw new IOException();
        }
        launch();
    }
}