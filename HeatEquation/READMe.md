# FDM in Heat Equation

> This repository implements a parallelized version of the FDM in Heat Equation proposed in the 14th Marathon of Parallel Programming presented by Universidade Mackenzie. The problem consists of a cube, that is heated on every face of it, and the goal of the algorithm is to define how many time steps does it takes so that the cube if fully heated., but the algorithm for that can be very expensive. As a alternative solution, a algorithm that runs on GPU, implemented here, tries to reduce this time significantly. More details can be found at the marathon specification, [Problem E](http://lspd.mackenzie.br/marathon/19/problemset.pdf).

<div align="center">
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtjpTbTOgJlXtUd6mvN5wYuthdscoXTNtYEA&usqp=CAU" alt="3D Grid" width="40%">
</div>

### ⚙️ Adjustments and improvements

No adjustments or improvements for the moment.

## 💻 Prerequisites

Before you start, check if you have matched the following requisites:
* Instal the following tools and their versions `< C \ CUDA >`.
* Windows as OS.

## 🚀 Compiling FDM in Heat Equation

To compile the program, simply run in command line:

Windows:
```
make
```

## ☕ Using FDM in Heat Equation

To use the project FDM in Heat Equation, simply run the executable:

- Windows:

  - Sequential code:

    ```
    mdf.exe
    ```

  - Cuda code:

    ```
    mdf_cuda.exe
    ```

A input test is available, the `fdm.in` file. It's possible to execute it by running the following command.

```
mdf.exe < fdm.in
```

The same for the cuda program.

## 🤝 Contributors

This project is presented by:

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/56005905?v=4" width="100px;" alt="Felipe Tavoni's profile pic on GitHub"/><br>
        <sub>
          <b>Felipe Tavoni</b>
        </sub>
      </a>
    </td>
  </tr>
</table>