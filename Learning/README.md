# Learning

Le but de cette partie est de montrer l'apprentissage possible de gaussienne en utilisant le langage Rust.
Les programmes de 3D Gaussian Splatting utilisent le module PyTorch pour gérer les données ainsi que la backpropagation de la loss.
Pour remplacer PyTorch en Rust, nous avons fait appel au framework [**Candle**](https://github.com/huggingface/candle.git) qui propose la gestion de Tensor comme dans PyTorch avec un certain nombre de fonction comme PyTorch, mais aussi la possibilité de backpropgation et l'appel de fonction CUDA.

Pour essayer cela, nous avons implémenté une version que nous pouvons appelé simplifié de 3DGS, [**gsplat**](https://github.com/nerfstudio-project/gsplat.git), par Nerfstudio qui reprend les principes des Gaussiennes de 3DGS mais pour une application plus 2D où le but est de seulement reconstruire une image. En arrivant à reconstruire une image à partir de Gaussiennes aléatoires, nous prouvons ainsi la possibilité de Learning à l'aide de l'outil Candle en Rust dans le cas de Gaussian Splatting.
