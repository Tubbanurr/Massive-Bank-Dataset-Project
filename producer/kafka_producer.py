"""
============================================================================
Kafka Producer - Bank Transaction Streaming
============================================================================
Bu script, CSV dosyasından banka işlem verilerini okur ve
gerçek zamanlı veri akışı simülasyonu yaparak Kafka topic'ine gönderir.

Her mesajda:
- timestamp: Mesajın gönderilme zamanı
- user_id: Simüle edilmiş kullanıcı ID'si
- event_type: İşlem tipi (DEPOSIT, WITHDRAWAL, TRANSFER, PAYMENT)
- Orijinal veri alanları (Date, Domain, Location, Value, Transaction_count)

Özellikler:
- Ayarlanabilir mesaj gönderim hızı (MESSAGE_RATE env variable)
- Batch gönderim desteği
- Producer logları ile kaç mesaj gönderildiği takip edilebilir
============================================================================
"""

import os
import csv
import json
import time
import random
import logging
from datetime import datetime
from kafka import KafkaProducer

# ===================== LOGGING AYARLARI =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('KafkaProducer')

# ===================== KONFIGÜRASYON =====================
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'bank-transactions')
MESSAGE_RATE = int(os.environ.get('MESSAGE_RATE', 50))  # mesaj/saniye
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 100))
CSV_FILE_PATH = '/app/dataset/bankdataset.csv'

# İşlem tipleri (simülasyon için)
EVENT_TYPES = ['DEPOSIT', 'WITHDRAWAL', 'TRANSFER', 'PAYMENT']


def create_kafka_producer(max_retries=30, retry_interval=5):
    """
    Kafka Producer oluşturur. Bağlantı kurulamazsa yeniden dener.
    
    Args:
        max_retries: Maksimum deneme sayısı
        retry_interval: Denemeler arası bekleme süresi (saniye)
    
    Returns:
        KafkaProducer instance
    """
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432,
                max_request_size=10485760
            )
            logger.info(f"✅ Kafka Producer bağlantısı başarılı! (Deneme {attempt})")
            logger.info(f"   Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")
            logger.info(f"   Topic: {KAFKA_TOPIC}")
            return producer
        except Exception as e:
            logger.warning(f"⏳ Kafka bağlantısı kurulamadı (Deneme {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_interval)
            else:
                logger.error("❌ Kafka bağlantısı kurulamadı! Tüm denemeler başarısız.")
                raise


def generate_user_id():
    """Rastgele kullanıcı ID'si üretir"""
    return f"USR-{random.randint(10000, 99999)}"


def enrich_message(row, row_index):
    """
    CSV satırını zenginleştirilmiş bir mesaja dönüştürür.
    
    Args:
        row: CSV satırı (dict)
        row_index: Satır numarası
    
    Returns:
        Zenginleştirilmiş mesaj (dict)
    """
    message = {
        # Meta bilgiler
        'message_id': row_index,
        'timestamp': datetime.now().isoformat(),
        'user_id': generate_user_id(),
        'event_type': random.choice(EVENT_TYPES),
        
        # Orijinal veri alanları
        'date': row.get('Date', ''),
        'domain': row.get('Domain', ''),
        'location': row.get('Location', ''),
        'value': int(row.get('Value', 0)),
        'transaction_count': int(row.get('Transaction_count', 0))
    }
    return message


def delivery_report(record_metadata):
    """Mesaj gönderim başarı callback'i"""
    logger.debug(
        f"📨 Mesaj gönderildi -> Topic: {record_metadata.topic}, "
        f"Partition: {record_metadata.partition}, "
        f"Offset: {record_metadata.offset}"
    )


def stream_csv_to_kafka():
    """
    Ana fonksiyon: CSV dosyasını okur ve Kafka'ya stream eder.
    
    Mesaj gönderim hızı MESSAGE_RATE (mesaj/saniye) ile kontrol edilir.
    Her 1000 mesajda bir log yazılır.
    """
    # Kafka Producer oluştur
    producer = create_kafka_producer()
    
    logger.info(f"📂 CSV dosyası okunuyor: {CSV_FILE_PATH}")
    logger.info(f"⚡ Mesaj hızı: {MESSAGE_RATE} mesaj/saniye")
    logger.info(f"📦 Batch boyutu: {BATCH_SIZE}")
    logger.info("=" * 60)
    
    total_sent = 0
    total_errors = 0
    start_time = time.time()
    batch_start_time = time.time()
    
    try:
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row_index, row in enumerate(reader, start=1):
                try:
                    # Mesajı zenginleştir
                    message = enrich_message(row, row_index)
                    
                    # Kafka'ya gönder
                    key = message['domain']  # Domain'e göre partitioning
                    future = producer.send(
                        KAFKA_TOPIC,
                        key=key,
                        value=message
                    )
                    future.add_callback(lambda m: delivery_report(m))
                    
                    total_sent += 1
                    
                    # Hız kontrolü
                    if total_sent % MESSAGE_RATE == 0:
                        elapsed = time.time() - batch_start_time
                        if elapsed < 1.0:
                            time.sleep(1.0 - elapsed)
                        batch_start_time = time.time()
                    
                    # Her 1000 mesajda log
                    if total_sent % 1000 == 0:
                        elapsed_total = time.time() - start_time
                        rate = total_sent / elapsed_total if elapsed_total > 0 else 0
                        logger.info(
                            f"📊 Gönderilen: {total_sent:,} mesaj | "
                            f"Hata: {total_errors} | "
                            f"Hız: {rate:.1f} mesaj/sn | "
                            f"Süre: {elapsed_total:.1f}s"
                        )
                    
                    # Her batch'te flush
                    if total_sent % BATCH_SIZE == 0:
                        producer.flush()
                
                except Exception as e:
                    total_errors += 1
                    logger.error(f"❌ Satır {row_index} gönderilemedi: {e}")
                    if total_errors > 100:
                        logger.error("🛑 Çok fazla hata! Durduruluyor.")
                        break
        
        # Son mesajları flush et
        producer.flush()
        
        # Özet
        elapsed_total = time.time() - start_time
        rate = total_sent / elapsed_total if elapsed_total > 0 else 0
        logger.info("=" * 60)
        logger.info("✅ STREAMING TAMAMLANDI!")
        logger.info(f"   Toplam gönderilen: {total_sent:,} mesaj")
        logger.info(f"   Toplam hata: {total_errors}")
        logger.info(f"   Toplam süre: {elapsed_total:.1f} saniye")
        logger.info(f"   Ortalama hız: {rate:.1f} mesaj/saniye")
        logger.info("=" * 60)
    
    except FileNotFoundError:
        logger.error(f"❌ CSV dosyası bulunamadı: {CSV_FILE_PATH}")
        raise
    except KeyboardInterrupt:
        logger.info("⚠️ Kullanıcı tarafından durduruldu.")
    finally:
        producer.close()
        logger.info("🔌 Kafka Producer kapatıldı.")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🏦 BANK TRANSACTION KAFKA PRODUCER")
    logger.info("=" * 60)
    stream_csv_to_kafka()
