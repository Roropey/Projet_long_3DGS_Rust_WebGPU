# Projet Long : 3D Gaussian Splatting en Rust et WebGPU
Clémentine Grethen, Jean-Félix Maestrati, Romain Peyremorte, Ghislain Réveiller

Ce dépot git correspond à un travail de recherche et test sur l'implémentation de [**3D Gaussian Splatting**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) en Rust. 
Il s'inscrit dans le cadre du Projet Long de fin d'étude à l'Ecole National Supérieur d'Electrotechnique, d'Electronique, d'Informatique, d'Hydraulique et des Télécommunication en partenariat avec Fittingbox.

Il est composé de deux parties s'intéressant aux possibilités de Rust dans le cadre de la partie *Learning* et la partie *Renderer*.
La partie *Learning* utilise le framework [**Candle**](https://github.com/huggingface/candle.git) avec Rust ainsi que CUDA pour l'interface GPU.
La partie *Renderer* utilise WebGPU pour l'interface GPU avec Rust.
