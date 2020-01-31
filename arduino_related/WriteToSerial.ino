
#include <MPU6050.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>


int shockPin = 3;
int tempPin = A0;

int pin1 = 2;
int pin2 = 3;
int pin3 = 4;
int pin4 = 5;
int pin5 = 6;
int pin6 = 7;
int pin7 = 8;
int pin8 = 9;
double inval = 0;
double serialD = 0;
MPU6050 accelerometer;

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 32 // OLED display height, in pixels

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_RESET     -1// Reset pin # (or -1 if sharing Arduino reset pin)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);




void setup() {
  pinMode(shockPin, INPUT);
  pinMode(tempPin, INPUT);
  pinMode(pin1, OUTPUT);
  pinMode(pin2, OUTPUT);
  pinMode(pin3, OUTPUT);
  pinMode(pin4, OUTPUT);
  pinMode(pin5, OUTPUT);
  pinMode(pin6, OUTPUT);
  pinMode(pin7, OUTPUT);
  pinMode(pin8, OUTPUT);
  pinMode(13, OUTPUT);

  while(!accelerometer.begin(MPU6050_SCALE_2000DPS, MPU6050_RANGE_2G)){
    Serial.print("??");
  }
  Serial.println("init achievied");

  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);


  Serial.begin(9600);

}

void loop() {

  if (Serial.available()){
    serialD = Serial.read();
    inval = serialD;
  }

  double output1 = (inval/5*8)+2;
  output1 = floor(output1);
  if (output1 > 9){
    output1 = 9;
  }
  for (int i = 2; i <= output1; i++){
    digitalWrite(i,HIGH);
  }
  for (int i = 10; i > output1; i--){
    digitalWrite(i,LOW);
  }
  
  Vector sensor_data = accelerometer.readNormalizeAccel();
  int pitch_value = -(atan2(sensor_data.XAxis, sqrt(sensor_data.YAxis*sensor_data.YAxis + sensor_data.ZAxis*sensor_data.ZAxis))*180.0)/M_PI;
  int roll_value = (atan2(sensor_data.YAxis, sensor_data.ZAxis)*180.0)/M_PI;

  String value = String(roll_value);

  inval = ((abs(pitch_value)/16)+(abs(roll_value)/35))/2;
  inval = 5-inval;
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.setTextColor(WHITE);
  display.clearDisplay();
  display.print("Driver Name: \n Rajesh");
  display.setCursor(0, 20);  
  display.print("Car safety: " + String(inval));
  display.display();

  delay(1000);

  display.clearDisplay();

  Serial.print(pitch_value);
  Serial.print(",");
  Serial.print(roll_value);
  Serial.print(",");
  Serial.println(analogRead(tempPin)-180);

  int data = 0;
  while(Serial.available()){
   data = Serial.read();
   Serial.println(data);
  }
//pins 3-10
  if (data = 1) {
    digitalWrite(pin1, HIGH);
  }
  else if (data = 2) {
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
  }
  else if(data = 3){
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
    digitalWrite(pin4, HIGH);
    digitalWrite(pin5, HIGH);
  }
  else if(data = 4){
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
    digitalWrite(pin4, HIGH);
    digitalWrite(pin5, HIGH);
    digitalWrite(pin6, HIGH);
    digitalWrite(pin7, HIGH);
    digitalWrite(pin8, HIGH);
  }
  else if(data = 5){

  }

}

void getData(){
 int actualData = 0;
  while(Serial.available()){
   int data = Serial.read();
   Serial.println(data);
    //actualData = (data, DEC);
   Serial.println(data, DEC);
  }
//pins 3-10
  if (actualData = 1) {
    digitalWrite(pin1, HIGH);
  }
  else if (actualData = 2) {
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
  }
  else if(actualData = 3){
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
    digitalWrite(pin4, HIGH);
    digitalWrite(pin5, HIGH);
  }
  else if(actualData = 4){
    digitalWrite(pin1, HIGH);
    digitalWrite(pin2, HIGH);
    digitalWrite(pin3, HIGH);
    digitalWrite(pin4, HIGH);
    digitalWrite(pin5, HIGH);
    digitalWrite(pin6, HIGH);
    digitalWrite(pin7, HIGH);
    digitalWrite(pin8, HIGH);
  }
  else if(actualData = 5){

  }
}
