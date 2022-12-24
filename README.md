# hifi-gan

В репозитории реализована и обучена модель HiFiGAN из статьи https://arxiv.org/pdf/2010.05646.pdf

Установка библиотек:
```
pip install -r requirements.py
```
Запустить обучение модели можно так:
```
python train.py
```
Чекпоинт предварительно нужно скачать отсюда https://drive.google.com/drive/folders/1yd76Yy7Rq0-MrkrSRZLyiiOO6WDczSvH?usp=sharing и поместить в папку из ```config.py```. В другом случае обучение будет проходить с нуля.

Данные загружаются в папку data, если их нет. Также автоматически модель будет обучаться начиная с чекпоинта и параметрами из ```config.py```.
В папке generated_wavs находятся сгенерированные аудиозаписи.

Ссылка на отчет wandb: https://wandb.ai/zzlatazz/dla-hw4-hifi/reports/HiFi-GAN--VmlldzozMjExNzkx
