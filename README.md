# Цифровой прорыв 2023: (Окружной хакатон 19-21 мая)

---
### Кейс: Центральный Банк России
*Описание*: <br>
На основе датасета обращений пользователей и результатов устранения нарушений участникам хакатона предлагается создать наукастинг-модель на базе методов машинного обучения, позволяющую выявлять всплески и периодические нарушения в работе ИТ-решений, устанавливать закономерности возникновения нарушений разных ИТ-решений, формировать перечень сбоев и прогноз их реализации в разрезе ИТ-решений.

### Решение команды GibData:
*Тизер*: <br>
Мы предоставляем модель для прогнозирования и поддержки стабильной работы ИТ-решений. Представленное решение позволяет минимизировать ошибки системы с помощью повышения качества классификации запросов засчет выявления в данных аномалий и причин их возникновения. С данным продуктом ЦБ РФ получит возможность более точной оценки потенциальных нарушений, которые могут быть как очевидными, нарушающими работу целого отдела, так и незначительными.

Используемые технологии в рамках реализации проекта: `Python`, `CatBoost`, `SBERT`, `LaBSE`, `Streamlit`.

Уникальность нашего решения заключается в создании почвы для автоматизации фиксаций нарушений в системе посредством реализации двух моделей с различными значениями целевых показателей: скорости и эффективности, при этом взаимодействие сотрудника с данной системой сводится к минимуму.

*Решение задачи №1*: <br>
Разработали два решения для бизнеса на выбор: 
1) Файл `hard_ml.ipynb` - тяжелое по ресурсам, затратное по времени, но точное по классификации (`CatBoost` + реализовано на нейросетевом подходе (`SBERT` от `SBER`) для выделения информации из текста)
2) Директория `web` - быстрое решение, которое можно эффективно интегрировать в бизнес, но с меньшей точностью классификации (реализовано с помощью Open-Source модели `CatBoost` от Yandex)

![color picker](https://s11.gifyu.com/images/outputvideo-cutter-js.com.gif) <br>
**Пример работы с веб-интерфейсом**
