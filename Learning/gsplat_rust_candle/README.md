# gsplat en Rust avec Candle

![Teaser](/gifs/lenagif.gif?raw=true)

Cette implémentation reprend les programmes [**gsplat**](https://github.com/nerfstudio-project/gsplat.git) de Nerfstudio mais en les implémentant en Rust et en appelant les programmes CUDA.
Certaines modifications ont été apporté en terme de type par rapport aux kernels CUDA et variables pour correspondre aux types utilisable avec Candle (par exemple le `int32_t` utilisé à l'origine dans les kernels de *rasterize* a été modifié en `int64_t`.

## Utilisation

Les programmes sont faits pour fonctionner avec un GPU Nvidia et CUDA. Il est donc nécessaire d'installer CUDA avant utilisation.
Il faut ensuite `build` avec la *feature* "cuda" :
```bash
cargo build --features cuda
```

Une fois `build`, un exécutable est disponible dans le dossier *target\debug* généré : `gsplat_rust_candle.exe`. Les options de cette exécutable sont :
- `--height` pour indiquer la hauteur de l'image voulu en traitement et de l'image de sortie (peut-être différent de la vrai hauteur de l'image d'entrée) (par défaut 256)
- `--width` pour indiquer la largeur de l'image voulu en traitement et de l'image de sortie (peut-être différent de la vrai largeur de l'image d'entrée) (par défaut 256)
- `--img-path` le chemin vers l'image à reproduire (par défaut image.png)
- `--save_imgs` l'indication pour ou non sauvegarder les résultats temporaires (par défaut true)
- `--iterations` le nombre d'itération à exécuter (par défaut 1000)
- `--num-points` le nombre de gaussienne qui évolue durant l'exécution (par défaut 100000)

Voici un exemple d'exécution :
```bash
.\target\debug\gsplat_rust_candle.exe --height 50 --width 50 --img-path lena.png --iterations 10000 --num-points 1000
```

## Architecture du dossier

À la racine, nous trouvons les fichiers `Cargo` qui indique les dépendances du project. Nous avons aussi `build.rs` qui est le programme permettant de trouver les programmes CUDA dans leur dossier et de réaliser le lien en Rust dans le fichier `cuda_kernels.rs`.
Nous avons aussi des images (*lena* et *circle*) qui nous ont servies durant nos tests.

Dans le dossier `src` nous retrouvons la majorité des programmes Rust similaires à [**gsplat**](https://github.com/nerfstudio-project/gsplat/tree/main/gsplat).
Dans le dossier `cuda`, nous retrouvons les programmes `customop.rs` et `bindings.rs` qui contiennent les fonctions importantes **ProjectGaussians** et **RasterizeGaussians** qui font appel aux programmes CUDA et gèrent les données ainsi que la backpropagation dans les programmes CUDA.
Le programme `cuda_kernels.rs` indique aux programmes Rust où trouver les programmes CUDA et lesquels sont appelables , c'est à dire dans le dossier `kernels` qui contient les fichiers *.cu* et *.cuh* ainsi que leurs dépendances.
