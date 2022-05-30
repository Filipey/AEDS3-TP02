# Problema de alocação em redes - Algoritmos e Estruturas de Dados III

## Participantes
- [Filipe Augusto Santos de Moura (Aluno)](https://github.com/Filipey)
- [Gustavo Estevam Sena (Aluno)](https://github.com/Gultes)
- [George Henrique Godim da Fonseca (Orientador)](https://github.com/georgehgfonseca)

## Objetivos
- Aplicar os conhecimentos em algoritmos para resolver um problema real.
- Aprimorar a habilidade de programação de algoritmos em grafos.
- Reforçar o aprendizado sobre os algoritmos de fluxo em redes.

## Sobre
O trabalho consiste em resolver o problema da alocação de professores às disciplinas do DECSI/UFOP
através de algoritmos de fluxo em redes. Cada professor leciona duas ou três disciplinas
e define, a cada semestre quais disciplinas tem preferência por lecionar dentre as que são ofertadas
pelo DECSI. Uma solução para esse problema consiste em uma atribuição de disciplinas aos professores
de modo a maximizar o atendimento de suas preferências. A entrada serão dois arquivos no formato
.csv (separado por vírgulas), um de professores e outro de disciplinas conforme o exemplo:

**professores.csv**

|   Professor    | Disciplinas | Preferência 1 | Preferência 2 | Preferência 3 |
|:--------------:|:-----------:|:-------------:|:-------------:|:-------------:|
| George Fonseca |      2      |    CSI105     |    CSI466     |    CSI601     |
| Bruno Monteiro |      3      |    CSI601     |    CSI602     |    CSI466     |

**disciplinas.csv**

| Disciplina |             Nome              | Turmas |
|:----------:|:-----------------------------:|:------:|
|   CSI105   | Alg. e Estrutura de Dados III |   1    |
|   CSI466   |       Teoria dos Grafos       |   1    |
|   CSI601   |       Banco de Dados I        |   2    |
|   CSI602   |       Banco de Dados II       |   1    |

O programa irá ler
esses arquivos de entrada e criar a rede de fluxo correspondente ao problema de alocação. A rede
de fluxo terá quatro camadas, um com o nó de super oferta, outra com nós representado os professores, outra
representando as disciplinas e, por fim, o nó de super demanda. Com relação às preferências, os seguintes
custtos incorrem:

| Preferência |  1  |  2  |  3  |  4  |  5  |
|:-----------:|:---:|:---:|:---:|:---:|:---:|
|    Custo    |  0  |  3  |  5  |  8  | 10  |

## Run 🏃‍

```bash
# Clone este repositório
$ git clone https://github.com/Filipey/AEDS3-TP02.git

# Acesse o diretório do projeto no terminal
$ cd AEDS3-TP02
````

No arquivo main.py, insira os seguintes dados:

```python
# Arquivo de professores no formato csv presente na pasta /dataset
teachers = input("Type the filename in /dataset: ")

# Arquivo de disciplinas no formato csv presente na pasta /dataset
subjects = input("Type the filename in /dataset: ")
```

A execução irá retornar no console a seguinte resposta:

|  Professor   | Disciplina |         Nome         |        Turmas        | Custo |
|:------------:|:----------:|:--------------------:|:--------------------:|:-----:|
| Professor #1 |   CSI###   | Nome da Disciplina 1 | Quantidade de Turmas |   X   |
| Professor #2 |   CSI###   | Nome da Disciplina 2 | Quantidade de Turmas |   X   |


## Feito com ❤️
