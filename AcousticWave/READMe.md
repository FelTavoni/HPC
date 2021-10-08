<h1 align="center"> 
    The propagation of a Acoustic Wave
</h1>

<h4 align="center">
    <img alt="RickerWave" title="#RickerWave" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/MexicanHatMathematica.svg/375px-MexicanHatMathematica.svg.png" width="300px;" />
</h4>

<p align="center">
	<a href="#About">About</a> |
	<a href="#How-to-execute">How to execute</a>
</p>

## 🔍 About

> The following code simulates the propagation of a acoustic wave (similar to the Ricker Wave). This propagation, depending on the size of the grid and the number of timesteps may take several minutes to compute. As that said, this projects maps it's processing to a GPU, decreasing the time that's needed to compute this propagation. A plot.py code shows the final form of the wave.

---

## 🔌 How to execute

### Prerequisites

Before you start, check if you've matched the following pre-requisites

[Python](https://www.python.org/downloads/)

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

PS: Windows was used as enviroment while developing this project. It's most likely to run on Linux too, but I have not tried it yet...

---

#### 💻 Compiling

To compile the available code, simply type on bash:

```bash
# For the sequential code
gcc wave_seq.c -o wave_seq.exe

# For the CUDA code
nvcc wave_cuda.c -o wave_cuda.exe
```

#### 🧭 Executing

To run the compiled code:

```bash
# For the sequential code
wave_seq.exe <number-of-lines> <number-of-cols> <number-of-timesteps>

# For the CUDA code
wave_cuda.exe <number-of-lines> <number-of-cols> <number-of-timesteps>
```

---

## 🛠 Technologies

The following tools were used in this project:

-   **[Python3](https://www.python.org/downloads/)**
-   **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

---

## 📘 References

- [Wikipedia - Ricker Wavelet](https://en.wikipedia.org/wiki/Ricker_wavelet)
- [Prof. Hermes Senger Discipline in UFSCar]()

## 🦸 Autor

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img style="border-radius: 25%" src="https://avatars.githubusercontent.com/u/56005905?v=4" width="100px;" alt="Foto de Felipe Tavoni"/><br>
        <sub>
          <b>Felipe Tavoni</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

---

<!-- ## 📝 Licença

Este projeto esta sobe a licença [MIT](./LICENSE).
 -->