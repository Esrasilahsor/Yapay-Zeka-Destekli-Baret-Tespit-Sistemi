/*
 * ==============================================================================
 * Proje: İş Sağlığı ve Güvenliği (İSG) Baret Denetim Sistemi - Arduino Uno R3
 * Açıklama: Python PyQt5 arayüzünden gelen seri port komutlarını ('1' / '0')
 *           dinler; Kırmızı/Yeşil LED ve Buzzer ikazlarını yönetir.
 *
 * Pin Konfigürasyonu:
 *   - Pin 8  : Yeşil LED (GÜVENLİ DURUM)
 *   - Pin 9  : Kırmızı LED (KRİTİK İHLAL / TEHLİKE)
 *   - Pin 10 : 5V Aktif Buzzer (SESLİ ALARM)
 *
 * Haberleşme Protokolü (9600 Baud):
 *   - '1' veya 'H': Baretsiz İhlal -> Kırmızı LED AÇIK, Buzzer Kesikli Ses, Yeşil LED KAPALI
 *   - '0' veya 'S': Güvenli Saha  -> Yeşil LED AÇIK, Kırmızı LED KAPALI, Buzzer KAPALI
 *   - 'C': Temizle / Bekleme      -> Tüm ikazlar KAPALI
 * ==============================================================================
 */

const int PIN_GREEN_LED = 8;
const int PIN_RED_LED   = 9;
const int PIN_BUZZER    = 10;

char currentCommand = '0';
unsigned long lastBuzzerToggle = 0;
bool buzzerState = false;

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_GREEN_LED, OUTPUT);
  pinMode(PIN_RED_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  
  // Başlangıçta tüm çıkışları sıfırla
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, LOW);
  digitalWrite(PIN_BUZZER, LOW);
}

void loop() {
  // Seri porttan yeni komut gelip gelmediğini kontrol et
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1' || cmd == '0' || cmd == 'H' || cmd == 'S' || cmd == 'C') {
      currentCommand = cmd;
    }
  }

  // Komut durumunu işlet
  if (currentCommand == '1' || currentCommand == 'H') {
    // Baretsiz İhlal Durumu: Kırmızı LED + Kesikli Buzzer
    digitalWrite(PIN_RED_LED, HIGH);
    digitalWrite(PIN_GREEN_LED, LOW);

    // Buzzer Kesikli İkaz (Düdük / Siren Efekti)
    unsigned long currentMillis = millis();
    if (currentMillis - lastBuzzerToggle >= 150) {
      lastBuzzerToggle = currentMillis;
      buzzerState = !buzzerState;
      digitalWrite(PIN_BUZZER, buzzerState ? HIGH : LOW);
    }
  } 
  else if (currentCommand == '0' || currentCommand == 'S') {
    // Güvenli Durum: Yeşil LED Aktif, Kırmızı ve Buzzer Pasif
    digitalWrite(PIN_GREEN_LED, HIGH);
    digitalWrite(PIN_RED_LED, LOW);
    digitalWrite(PIN_BUZZER, LOW);
    buzzerState = false;
  } 
  else {
    // Bekleme veya Kapalı Durum ('C')
    digitalWrite(PIN_GREEN_LED, LOW);
    digitalWrite(PIN_RED_LED, LOW);
    digitalWrite(PIN_BUZZER, LOW);
    buzzerState = false;
  }
}
