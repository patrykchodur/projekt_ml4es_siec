# projekt_ml4es_siec

Aby ściągnąć dataset i przetrenować sieć należy uruchomić skrypt run_all.sh
Rezultatem jego działania jest plik model_converted.tflite, który należy umieścić
w katalogu assets w projekcie Android Studio. Na koniec dostajemy ponadto
2 pliki tekstowe: test_names.txt - zawiera ścieżki zdjęć wykorzystanych przy
walidacji, oraz train_names.txt - zawiera ścieżki zdjęć wykorzystanych przy
uczeniu sieci. Skrypt zostawia jeszcze foldery z oryginalnymi zdjęciami oraz
przyciętymi do rozmiaru 224x224 (\*\_cut).

Do działania skrypt wymaga MagicImage, wget, python3 i tensorflow wraz z keras.
