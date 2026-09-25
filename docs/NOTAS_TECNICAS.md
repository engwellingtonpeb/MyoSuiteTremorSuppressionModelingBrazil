# Notas técnicas: decisões, diagnósticos e resultados parciais

O ponto de partida é a tese (MATLAB/OpenSim, `DSc/07_TremorModel_v5`), cujo fluxo é
planta → identificação DMDc → síntese H∞ (`mixsyn`) → malha fechada, sem o oscilador de Matsuoka.
Aqui tudo foi reimplementado em Python nativo sobre o MyoSuite/MuJoCo.

## Correspondência entre os módulos (`wrist_control/`) e a tese

| módulo | papel | equivalente na tese |
|---|---|---|
| `plant.py` | myoArm reduzido na postura do `.osim` (ombro/cotovelo travados e "embutidos"), dt = 1 ms; `WristPlant` (2 GDL, 7 músculos) e `WristPlantND("3dof")` (3 GDL, 8 músculos) | `.osim` + `OsimPlantFcn.m` + `FirstOrderActivationDynamics.m` |
| `sysid.py` | experimentos de identificação, DMDc afim, SINDYc estruturado, validação | `idtf_signals.m`, `dmdc_3.m`, `DelayDMDc_MV.m` |
| `sindyc.py` | STLSQ/STRidge (porta de `sparsifyDynamics.m`) e o estudo SINDYc × DMDc em 2 GDL | `04_DMDc_IDTF/utils` |
| `synthesis.py` | d2c exato, `makeweight`, `mixsyn`, residualização, c2d, equilíbrio e linearização | `ControllerSynthesis.m` |
| `closed_loop.py` | controlador discreto, inibição recíproca (2 GDL), `ControllerND` com projeção no espaço nulo | `OsimControlsFcn.m` |
| `pipeline_*.py` | fluxos completos de cada estudo | `main.m` |

A ativação muscular usa a dinâmica nativa do MuJoCo, que tem a mesma forma de `FirstOrderActivationDynamics.m`:
τ·(0,5 + 1,5a), com τ_act = 12 ms e τ_deact = 40 ms.

---

## Estudo 1: punho em 2 GDL, 7 músculos (`scripts/01`, `scripts/02` → `results/wrist2dof/`)

- **Canais de performance (θ, φ).** A formulação da tese pondera S nos 8 estados com apenas 4 entradas. Isso é
  infactível: ‖W1·S‖∞ ≥ 10, e a norma verificada foi ≈ 106, embora o slycot tenha reportado γ = 0,92. Provavelmente é
  por isso que a tese precisava de ganhos k ≈ 1e6. A formulação antiga ainda está disponível com `perf_idx=None`.
- **W2 = 0,01·I** em vez de `[]`, porque o slycot exige D12 de posto completo.
- **DMDc afim** (x⁺ = Ax + Bu + d), para incluir a gravidade. O controlador atua em torno do equilíbrio:
  u = u* + K(z)·(r − x).
- **Residualização** dos polos do controlador acima de 2/Ts; o `mixsyn` gera um polo de ~1e9 rad/s.
- O γ é **conferido numericamente** a partir de S, KS e T.
- Resultado: γ = 0,875; RMSE de regulação de 0,4° / 0,2°; RMSE nos degraus de ±10° de 3,1° / 2,2°
  (2,1° / 1,9° com inibição recíproca).
- SINDYc × DMDc (validação livre): θ passa de 52% a 96%, e o fit médio de 73% a 98%.

## Estudo 2 [histórico]: 3 GDL, 8 músculos, φ = 20°, identificação em malha fechada (`scripts/03`, `04`)

Músculos: SUP, ECRL, ECRB, ECU, FCR, FCU, PQ e **PT**. O PT dobra o torque mínimo que o conjunto gera em
qualquer direção 3D, de 2,1 para 4,1 Nm. Achados:
- Com dados de **pequena amplitude**, os termos a, a·q e a·q² da biblioteca ficam colineares. Isso exigiu três ajustes:
  biblioteca centrada em q_op, limiar pela contribuição de cada termo ao alvo e ridge.
- O SINDYc **podava a rigidez**, porque ela contribui pouco para Δq̇ em um passo de 1 ms. A solução foi proteger a
  parte linear [1, q, q̇, a] nas equações de aceleração. Com isso a rigidez identificada bate com o MuJoCo (~15%).
- **Espaço nulo de torque:** 8 músculos para 3 torques deixam 5 direções que só mudam a co-contração. O H∞
  deixava a co-contração derivar até saturar em 0. A projeção δu ← (I − NNᵀ)δu resolveu isso, sem alterar a malha linear.
- Co-contração basal u_base = 0,15 no equilíbrio.
- DMDc × SINDYc com a mesma síntese (`compare_dmdc_sindyc.log`): desempenho em malha fechada parecido.
  O SINDYc usa 78 coeficientes contra 322, rastreia melhor em regime e é mais lento para calcular o equilíbrio.

## Estudo 3 [ATUAL]: 3 GDL, palma para baixo, identificação em malha aberta (`scripts/05`, `06`)

Ponto de operação θ = 0°, ψ = 0°, φ = 80°. Mão e antebraço horizontais (±1°) e palma voltada para o solo, com
13° de resíduo. Chegar mais perto exigiria φ = 90°, que é o limite da junta.

1. **Identificação SINDYc em malha aberta.** A planta é **instável em malha aberta** nessa postura, para qualquer
   co-contração (σ ≈ +1,3 s⁻¹, modo flexão–desvio, deriva de ~20° em 3 s). Por isso:
   - o PD+I serve *só* para encontrar o equilíbrio u*;
   - depois são rodados 60 episódios de 1,5 s em **malha aberta pura**, u = u* + multisenoide independente por músculo
     (1–15 Hz) + PRBS, sempre a partir do mesmo estado;
   - a validação é por simulação livre em malha aberta dos episódios fora do treino, com esparsificação até
     min(fit) ≥ 80%.
   - Resultado: 125 termos e **fit mínimo de 93,4%** (DMDc: 62%). O modelo recuperou o polo instável: **+1,42 s⁻¹, contra +1,34 s⁻¹
     no MuJoCo**.
2. **Síntese:** equilíbrio e linearização do SINDYc, `mixsyn` com W1/W3 da tese (γ = 0,91), polos sobre o eixo jω
   deslocados de ε = 0,01 rad/s, pré-alimentação u*(r) linearizada e projeção no espaço nulo.
3. **Rastreamento:** RMSE nos degraus (incluindo um salto de 10° em θ) de 1,1° / 0,7° / 0,6°; nas senoides de 0,3–0,5 Hz,
   0,5° / 0,2° / 0,2°.
4. **Tremor de 5 Hz:** excitação alternada em meia-onda (flexores × extensores, pronadores × SUP), com amplitude de 0,08,
   somada ao comando enquanto o controlador mantém a postura. Na planta: 1,8° pico a pico em θ, com pico espectral
   em 5,0 Hz (harmônicos em 10 e 15 Hz).

### Artefato da geometria do myoArm perto da pronação máxima
- O tendão do **SUP troca de lado no wrapping em φ = 75°**. O braço de momento em φ salta de −5,9 para +6,4 mm, e acima
  de 75° o supinador passa a pronar.
- O PT e o BIC invertem o sinal gradualmente entre 65° e 75°.
- Consequência: a malha divergia sempre que φ cruzava 75°. Por isso o rastreamento mantém **φ entre 77° e 83°**.
- Mudar o `sidesite` (para dentro ou para fora do cilindro) ou o raio do cilindro **não** corrigiu o problema.

**Opções em aberto:**
- (a) manter φ = 80° com φ ≥ 76° (situação atual);
- (b) usar φ_op ≈ 65–70°, com a palma a 23–28° da horizontal;
- (c) refazer a geometria dos tendões do SUP e do PT.

### Outras pendências
- Com co-contração de 0,15, a rigidez estimada no estudo 2 errava 65–75%. A causa é que os dados foram gerados com
  outra co-contração. No estudo 3 a identificação já usa a co-contração de operação.
- O visualizador interativo (`scripts/06 --interactive`) não foi testado, porque precisa de janela.
