#include <iostream>
#include "nmmintrin.h" // for SSE4.2
#include <cstdlib> // для system
using namespace std;


__m128 initialize_register(float f) {
    return  _mm_set_ps(f, 0, 0, 0);   
}

__m128 count_discriminant(__m128 a, __m128 b, __m128 c) {
    __m128 b_square = _mm_mul_ps(b, b);
    __m128 four_ac = _mm_mul_ps(initialize_register(4), _mm_mul_ps(a, c));
    return _mm_sub_ps(b_square, four_ac);
}

int is_negative(__m128 n) {
    __m128 mask =  _mm_cmplt_ps(n, _mm_setzero_ps()); 
    int result = _mm_movemask_ps(mask); // return 0...0100 if n[3] < 0
    return result & 8;
}


__m128 count_x(__m128 a, __m128 b, __m128 discriminant, __m128 (*f)(__m128, __m128)) {
    __m128 sqr_d = _mm_sqrt_ps(discriminant);
    __m128 minus_b = _mm_sub_ps(_mm_setzero_ps(), b);
    __m128 quotient = f(minus_b, sqr_d);
    __m128 four_mul_a = _mm_mul_ps(initialize_register(2), a);
    return _mm_div_ps(quotient, four_mul_a);
}

int main() {   
    cout << "input coefficients: \n";
    float av, bv, cv;
    cin >> av >> bv >> cv;
    __m128 a = initialize_register(av);
    __m128 b = initialize_register(bv);
    __m128 c = initialize_register(cv);
    __m128 discriminant = count_discriminant(a, b, c);
   
    if (is_negative(discriminant)) {
       cout << "there is no roots";
       return 1;
    }
    
    __m128 x_one = count_x(a, b, discriminant,[](__m128 x, __m128 y) -> __m128 {return _mm_sub_ps(x, y);});
    __m128 x_two = count_x(a, b, discriminant, [](__m128 x, __m128 y) -> __m128 {return _mm_add_ps(x, y);});
   
   cout << "x_one: " << x_one[3] << "\n";
   cout << "x_two: " << x_two[3] << "\n";


    int z;
    system("pause"); // Только для тех, у кого MS Visual Studio
    return 0; 
}
