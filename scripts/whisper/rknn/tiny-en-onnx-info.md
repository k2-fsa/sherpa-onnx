# tiny.en encoder

```
----input----
NodeArg(name='tiny.en-mel', type='tensor(float)', shape=[1, 80, 3000])

-----output-----
NodeArg(name='cross_k_0', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_v_0', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_k_1', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_v_1', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_k_2', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_v_2', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_k_3', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='cross_v_3', type='tensor(float)', shape=[1, 1500, 384])
```

# tiny.en decoder

```
----input----
NodeArg(name='tiny.en-tokens', type='tensor(int32)', shape=[1, 1])
NodeArg(name='tiny.en-self_k_0', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_v_0', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_k_1', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_v_1', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_k_2', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_v_2', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_k_3', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-self_v_3', type='tensor(float)', shape=[1, 448, 384])
NodeArg(name='tiny.en-cross_k_0', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_v_0', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_k_1', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_v_1', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_k_2', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_v_2', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_k_3', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-cross_v_3', type='tensor(float)', shape=[1, 1500, 384])
NodeArg(name='tiny.en-offset', type='tensor(int32)', shape=[1])
NodeArg(name='tiny.en-mask', type='tensor(int32)', shape=[448])

-----output-----

NodeArg(name='tiny.en-logits', type='tensor(float)', shape=[1, 1, 51864])
NodeArg(name='tiny.en-this_self_k_0', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_v_0', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_k_1', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_v_1', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_k_2', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_v_2', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_k_3', type='tensor(float)', shape=[1, 1, 384])
NodeArg(name='tiny.en-this_self_v_3', type='tensor(float)', shape=[1, 1, 384])
```
