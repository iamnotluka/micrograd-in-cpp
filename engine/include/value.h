#pragma once

class Value {
    public:
        Value(double data);
        double data() const;

    private:
        double data_;
};