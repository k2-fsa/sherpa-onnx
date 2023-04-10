/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpaonnx;

public class EndpointConfig {
    final private EndpointRule rule1;
    final private EndpointRule rule2;
    final private EndpointRule rule3;

    public EndpointConfig(EndpointRule rule1, EndpointRule rule2, EndpointRule rule3) {
        this.rule1 = rule1;
        this.rule2 = rule2;
        this.rule3 = rule3;
    }

    public EndpointRule getRule1() {
        return rule1;
    }

    public EndpointRule getRule2() {
        return rule2;
    }

    public EndpointRule getRule3() {
        return rule3;
    }
}
