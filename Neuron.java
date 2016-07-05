import java.util.Vector;
import java.util.Random;


public class Neuron {

    private Double value;
    private Double h;
    private boolean biasNeuron;
    private Vector<Double> weights;
    private Vector<Double> previousWeights;

    //CONSTRUCTORS
    public Neuron() {
        this.value = 0.0;
        this.h = 0.0;
        this.biasNeuron = false;
        this.weights = new Vector<Double>();
        this.previousWeights = new Vector<Double>();
    }

    public Neuron(boolean bias) {
        this.value = 0.0;
        this.h = 0.0;
        this.biasNeuron = bias;
        this.weights = new Vector<Double>();
        this.previousWeights = new Vector<Double>();
    }

    public Neuron(int numConnections) {
        this.value = 0.0;
        this.h = 0.0;
        this.biasNeuron = false;
        this.weights = new Vector<Double>();
        this.previousWeights = new Vector<Double>();
        Random randNum = new Random();
        for(int i=0; i<numConnections; i++) {
            this.weights.add(randNum.nextDouble());
            this.previousWeights.add(0.0);
        }
    }

    public Neuron(boolean bias, int numConnections) {
        this.value = 0.0;
        this.h = 0.0;
        this.biasNeuron = bias;
        this.weights = new Vector<Double>();
        this.previousWeights = new Vector<Double>();
        Random randNum = new Random();
        for(int i=0; i<numConnections; i++) {
            this.weights.add(randNum.nextDouble());
            this.previousWeights.add(0.0);
        }
    }

    //METHODS
    public Double getDiffWeight(int pos) {
        return (this.weights.get(pos) - this.previousWeights.get(pos));
    }

    public Vector<Double> getDiffAllWeights() {
        Vector<Double> diffWeights = new Vector();
        for(int i=0; i<this.weights.size(); i++) {
            diffWeights.add(this.weights.get(i) - this.previousWeights.get(i));
        }
        return diffWeights;
    }

    public void print() {
        System.out.println("V: " + this.value + " H: " + this.h);
    }

    public void calculateH(int f, Double l) {
        switch (f) {
            case 1:
                this.h = 1/(1 + (Math.exp( ((-1)*l)*this.value )) );
                break;
        }
    }

    //SETTERS
    public void setValue(Double v) {
        this.value = v;
    }

    public void setH(Double newH) {
        this.h = newH;
    }

    public void setBiasNeuron(boolean bias) {
        this.biasNeuron = bias;
    }

    public void setWeights(Vector<Double> w) {
        this.weights = w;
    }

    public void setPreviousWeights(Vector<Double> prevW) {
        this.previousWeights = prevW;
    }

    //GETTERS
    public Double getValue() {
        return this.value;
    }

    public Double getH() {
        return this.h;
    }

    public boolean getBiasNeuron() {
        return this.biasNeuron;
    }

    public Vector<Double> getWeights() {
        return this.weights;
    }

    public Vector<Double> getPreviousWeights() {
        return this.previousWeights;
    }

    public Double getWeight(int pos) {
        return this.weights.get(pos);
    }
}
