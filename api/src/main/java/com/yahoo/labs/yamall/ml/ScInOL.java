// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.labs.yamall.ml;

import java.util.Arrays;

import com.yahoo.labs.yamall.core.Instance;
import com.yahoo.labs.yamall.core.SparseVector;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * Scale Invariant Online Learning (ScInOL).
 * <p>
 * The details of the algorithm are from
 * <p>
 * Michał Kempka, Wojciech Kotłowski, Manfred K. Warmuth
 * "Adaptive scale-invariant online algorithms for learning linear models" ICML 2019
 * <p>
 *
 * @author Michal Kempka
 * @version 0.1
 */
@SuppressWarnings("serial")
public class ScInOL implements Learner {
    private static final double SMALL_NUMBER = 1e-15;
    private transient double[] w = null;
    private transient double[] sumGradient;
    private transient double[] sumSquaredGradient;
    private transient double[] maxFeature;
    private transient double[] eta;

    private boolean clipGradient;
    private Loss lossFnc;
    private double L = 1;
    private int sizeHash;

    public ScInOL(int bits, double L, double epsilon) {
        this.sizeHash = 1 << bits;
        this.w = new double[this.sizeHash];
        sumGradient = new double[this.sizeHash];
        sumSquaredGradient = new double[this.sizeHash];
        maxFeature = new double[this.sizeHash];
        Arrays.fill(maxFeature, SMALL_NUMBER);
        eta = new double[this.sizeHash];
        this.L = L;
        this.setLearningRate(epsilon);
        this.clipGradient = true;
    }

    public ScInOL(int bits) {
        this(bits, 1.0, 1.0);
    }


    public void setClipGradient(boolean f)
    {
        this.clipGradient = f;
    }
    public double update(Instance sample) {
        double pred = 0;
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            final int i = entry.getIntKey();
            final double x_i = entry.getDoubleValue();
            maxFeature[i] = Math.max(maxFeature[i], L * Math.abs(x_i));

            final double g_i = sumGradient[i];
            final double s2_i = sumSquaredGradient[i];
            final double m_i = maxFeature[i];
            final double eta_i = eta[i];
            final double denominator = Math.sqrt(s2_i + m_i * m_i);
            final double theta_i = g_i / denominator;
            this.w[i] = Math.signum(theta_i) * Math.min(1, Math.abs(theta_i)) / (2 * denominator) * eta_i;

            pred += x_i * this.w[i];
        }

        double negativeGrad = this.lossFnc.negativeGradient(pred, sample.getLabel(), sample.getWeight());
        if (this.clipGradient) {
            negativeGrad = Math.min(negativeGrad,L);
            negativeGrad = Math.max(negativeGrad,-L);
        }
        for (Int2DoubleMap.Entry entry : sample.getVector().int2DoubleEntrySet()) {
            final int key = entry.getIntKey();
            final double x_i = entry.getDoubleValue();
            this.sumGradient[key] += x_i * negativeGrad;
            this.sumSquaredGradient[key] += Math.pow(x_i * negativeGrad, 2);
            this.eta[key] += x_i * negativeGrad * this.w[key];
        }

        return pred;
    }

    public double predict(Instance sample) {
        return sample.getVector().dot(this.w);
    }

    public void setLoss(Loss lossFnc) {
        this.lossFnc = lossFnc;
    }

    public void setLearningRate(double eta) {
        for (int i = 0; i < this.eta.length; ++i) {
            this.eta[i] = eta;
        }
    }

    public Loss getLoss() {
        return this.lossFnc;
    }

    public SparseVector getWeights() {
        return SparseVector.dense2Sparse(w);
    }


    public String toString() {
        String tmp = "Using ScInOL\n";
        tmp = tmp + "Loss function = " + getLoss().toString();
        return tmp;
    }

    public void setL(double L) {
        this.L = L;
    }

    private void writeObject(ObjectOutputStream o) throws IOException {
        o.defaultWriteObject();
        o.writeObject(SparseVector.dense2Sparse(w));
        o.writeObject(SparseVector.dense2Sparse(sumGradient));
        o.writeObject(SparseVector.dense2Sparse(sumSquaredGradient));
        o.writeObject(SparseVector.dense2Sparse(maxFeature));
        o.writeObject(SparseVector.dense2Sparse(eta));
    }

    private void readObject(ObjectInputStream o) throws IOException, ClassNotFoundException {
        o.defaultReadObject();
        this.w = ((SparseVector) o.readObject()).toDenseVector(sizeHash);
        this.sumGradient = ((SparseVector) o.readObject()).toDenseVector(sizeHash);
        this.sumSquaredGradient = ((SparseVector) o.readObject()).toDenseVector(sizeHash);
        this.maxFeature = ((SparseVector) o.readObject()).toDenseVector(sizeHash);
        this.eta = ((SparseVector) o.readObject()).toDenseVector(sizeHash);
    }
}
