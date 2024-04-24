// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class EndpointConfig {

    private final EndpointRule rule1;
    private final EndpointRule rule2;
    private final EndpointRule rule3;

    private EndpointConfig(Builder builder) {
        this.rule1 = builder.rule1;
        this.rule2 = builder.rule2;
        this.rule3 = builder.rule3;
    }

    public static Builder builder() {
        return new Builder();
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

    public static class Builder {

        private EndpointRule rule1 = EndpointRule.builder().
                setMustContainNonSilence(false).
                setMinTrailingSilence(2.4f).
                setMinUtteranceLength(0).
                build();
        private EndpointRule rule2 = EndpointRule.builder().
                setMustContainNonSilence(true).
                setMinTrailingSilence(1.4f).
                setMinUtteranceLength(0).
                build();
        private EndpointRule rule3 = EndpointRule.builder().
                setMustContainNonSilence(false).
                setMinTrailingSilence(0.0f).
                setMinUtteranceLength(20.0f).
                build();

        public EndpointConfig build() {
            return new EndpointConfig(this);
        }

        public Builder setRule1(EndpointRule rule) {
            this.rule1 = rule;
            return this;
        }

        public Builder setRule2(EndpointRule rule) {
            this.rule2 = rule;
            return this;
        }

        public Builder setRul3(EndpointRule rule) {
            this.rule3 = rule;
            return this;
        }
    }
}
