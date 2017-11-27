#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
// for char_bit
#include "GL/glew.h"
#include <limits.h>

#include "omp.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

using namespace std;
using vArray = vector<float>;

const char*
openGLErrorString(GLenum _errorCode)
{
  // Only 3.2+ Core and ES 2.0+ errors, no deprecated strings like stack
  // underflow etc.
  if (_errorCode == GL_INVALID_ENUM) {
    return "GL_INVALID_ENUM";
  } else if (_errorCode == GL_INVALID_VALUE) {
    return "GL_INVALID_VALUE";
  } else if (_errorCode == GL_INVALID_OPERATION) {
    return "GL_INVALID_OPERATION";
  } else if (_errorCode == GL_INVALID_FRAMEBUFFER_OPERATION) {
    return "GL_INVALID_FRAMEBUFFER_OPERATION";
  } else if (_errorCode == GL_OUT_OF_MEMORY) {
    return "GL_OUT_OF_MEMORY";
  } else if (_errorCode == GL_NO_ERROR) {
    return "GL_NO_ERROR";
  } else {
    return "unknown error";
  }
}

void
printShaderLog(GLuint shader)
{
  // Make sure name is shader
  if (glIsShader(shader)) {
    // Shader log length
    int infoLogLength = 0;
    int maxLength = infoLogLength;

    // Get info string length
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);

    // Allocate string
    char* infoLog = new char[maxLength];

    // Get info log
    glGetShaderInfoLog(shader, maxLength, &infoLogLength, infoLog);
    if (infoLogLength > 0) {
      // Print Log
      printf("%s\n", infoLog);
    }

    // Deallocate string
    delete[] infoLog;
  }

  else {
    printf("Name %d is not a shader\n", shader);
  }
}

void
CheckGLError(std::string str)
{
  GLenum error = glGetError();
  if (error != GL_NO_ERROR) {
    printf("Error! %s %s\n", str.c_str(), openGLErrorString(error));
  }
}

bool
checkShader(GLuint shaderIn, string shaderName)
{
  glCompileShader(shaderIn);
  // Check fragment shader for errors
  GLint fShaderCompiled = GL_FALSE;
  glGetShaderiv(shaderIn, GL_COMPILE_STATUS, &fShaderCompiled);
  if (fShaderCompiled != GL_TRUE) {
    cout << "Unable to compile " << shaderName << " shader " << shaderIn
         << "\n";
    printShaderLog(shaderIn);
    return false;
  }
  CheckGLError("shaders compiled");
}
inline int
array2Dto1D(int x, int y, int dim)
{
  return x + y * dim;
}
float
ImplicitCircle(int x, int y, int radius)
{
  float X = x * x;
  float Y = y * y;

  float ret = sqrt(X + Y) - radius;

  return ret;
}

float
mix(float x, float y, float alpha)
{
  return x * (1.0f - alpha) + (y * alpha);
}

inline float
fastModf(float in, float& inpart)
{
  // floor or int? floor because of negatives
  inpart = floor(in);
  return (in - inpart);
}

float
sampleUExact(const vector<float>& uArray, int u, int v, int dim)
{
  return (uArray[array2Dto1D(u, v, dim)] + uArray[array2Dto1D(u + 1, v, dim)]) /
         2.0f;
}

float
sampleVExact(const vector<float>& vArray, int u, int v, int dim)
{
  return (vArray[array2Dto1D(u, v, dim)] + vArray[array2Dto1D(u, v + 1, dim)]) /
         2.0f;
}

float
sampleUAtFaceU(const vector<float>& uArray, int u, int v, int dim)
{
  return uArray[array2Dto1D(u, v, dim)];
}

float
sampleVAtFaceU(const vector<float>& vArray, int u, int v, int dim)
{
  return (vArray[array2Dto1D(u, v - 1, dim)] +
          vArray[array2Dto1D(u + 1, v - 1, dim)] +
          vArray[array2Dto1D(u , v, dim)] + vArray[array2Dto1D(u + 1, v, dim)]) /
         4.0f;
}



float
sampleUAtFaceV(const vector<float>& uArray, int u, int v, int dim)
{
  return (uArray[array2Dto1D(u, v - 1, dim)] +
          uArray[array2Dto1D(u + 1, v - 1, dim)] +
          uArray[array2Dto1D(u, v, dim)] + uArray[array2Dto1D(u + 1, v, dim)]) /
         4.0f;
}



float
sampleVAtFaceV(const vector<float>& vArray, int u, int v, int dim)
{
  return vArray[array2Dto1D(u, v, dim)];
}

float
sampleTrilinear(const vector<float>& arrayIn, float u, float v, int dim)
{
  float uintpart, vintpart;

  // this does
  float floatPartU = fastModf(u, uintpart);
  float floatPartV = fastModf(v, vintpart);

//  float floatPartU = std::modf(u, &uintpart);
//  float floatPartV = std::modf(v, &vintpart);

  float tmp1{ 0.0f }, tmp2{ 0.0f }, tmp3{ 0.0f }, tmp4{ 0.0f }, tmp5{ 0.0f },
    tmp12{ 0.0f }, tmp34{ 0.0f };

  tmp1 = arrayIn[array2Dto1D((int)uintpart, (int)vintpart, dim)];
  tmp2 = arrayIn[array2Dto1D((int)uintpart + 1, (int)vintpart, dim)];
  tmp3 = arrayIn[array2Dto1D((int)uintpart, (int)vintpart + 1, dim)];
  tmp4 = arrayIn[array2Dto1D((int)uintpart + 1, (int)vintpart + 1, dim)];
  tmp12 = mix(tmp1, tmp2, floatPartU);
  tmp34 = mix(tmp3, tmp4, floatPartU);

  return mix(tmp12, tmp34, floatPartV);
}

template <typename T>
T
sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

template <typename T>
inline T
when_eq(T x, T y)
{
  return 1.0 - abs(sign(x - y));
}

template <typename T>
inline T
when_neq(T x, T y)
{
  return abs(sign(x - y));
}

template <typename T>
inline T
when_gt(T x, T y)
{
  return max(sign(x - y), (T)(0));
}

template <typename T>
inline T
when_lt(T x, T y)
{
  return max(sign(y - x), (T)(0));
}

template <typename T>
inline T
when_ge(T x, T y)
{
  return 1.0 - when_lt(x, y);
}

template <typename T>
inline T
when_le(T x, T y)
{
  return 1.0 - when_gt(x, y);
}

template <typename T>
inline T
or_(T a, T b)
{
  return min(a + b, 1.0f);
}

void
simpleAdvect(const vArray& u, const vArray& v, vArray& densityWrite,
             const vArray& densityRead, const int dimension)
{
  int i;
  int j;

#pragma omp parallel for
  for (j = 1; j < dimension-1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension-1; i++) {
      // float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension + 1);
      // float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension + 1);

      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i - uComponent;
      float vOffset = j - vComponent;

      float outsideleft = when_lt(uOffset, 1.0f);
      float outsideright = when_gt(uOffset, (float)(dimension - 2));
      float outsidebottom = when_lt(vOffset, 1.0f);
      float outsidetop = when_gt(vOffset, (float)(dimension - 2));

      float finalOutside =
        or_(or_(or_(outsideleft, outsideright), outsidebottom), outsidetop);

      // boundary conditions, other wise in trilinear sample, array2dto1d
      // integer overflows and it samples the other side.
      // also prevents out of bounds sampling.
      uOffset = std::min((float)(dimension - 2), std::max(1.0f, uOffset));
      vOffset = std::min((float)(dimension - 2), std::max(1.0f, vOffset));

      // blend based on if outside or not. might need to disable this once
      // proper BC are met in pressure solve. min/max is fine though.
      //      float densitySample =
      //        mix(sampleTrilinear(densityRead, uOffset, vOffset, dimension),
      //        0.0f,
      //            finalOutside);
      float densitySample =
        sampleTrilinear(densityRead, uOffset, vOffset, dimension);
      densityWrite[array2Dto1D(i, j, dimension)] = densitySample;
    }
  }
}

void
macCormackAdvect(const vArray& u, const vArray& v, vArray& densityWrite,
                 vArray& tempDensity, const vArray& densityRead,
                 const int dimension)
{
  int i;
  int j;

#pragma omp parallel for
  for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension - 1; i++) {
      //            float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension
      //            + 1);
      //            float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension
      //            + 1);

      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i - uComponent;
      float vOffset = j - vComponent;

      // boundary conditions, other wise in trilinear sample, array2dto1d
      // integer overflows and it samples the other side.
      // also prevents out of bounds sampling.
      uOffset = std::min((float)(dimension - 2), std::max(1.0f, uOffset));
      vOffset = std::min((float)(dimension - 2), std::max(1.0f, vOffset));

      // float densitySample = mix( sampleTrilinear(densityRead, uOffset,
      // vOffset, dimension), 0.0f, finalOutside);
      float densitySample =
        sampleTrilinear(densityRead, uOffset, vOffset, dimension);

      // tempDensity[array2Dto1D(i, j, dimension)] = mix(densitySample, 0.0f,
      // finalOutside);
      tempDensity[array2Dto1D(i, j, dimension)] = densitySample;
    }
  }
#pragma omp barrier

#pragma omp parallel for
  for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension - 1; i++) {
      //      float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension + 1);
      //      float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension + 1);
      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i + uComponent;
      float vOffset = j + vComponent;

      float OlduOffset = i - uComponent;
      float OldvOffset = j - vComponent;

      float outsideleft = when_lt(OlduOffset, 1.0f);
      float outsideright = when_gt(OlduOffset, (float)(dimension - 2));
      float outsidebottom = when_lt(OldvOffset, 1.0f);
      float outsidetop = when_gt(OldvOffset, (float)(dimension - 2));

      float finalOutside =
        or_(or_(or_(outsideleft, outsideright), outsidebottom), outsidetop);

      //            uOffset = std::min((float)(dimension - 1), std::max(0.0f,
      //            uOffset));
      //            vOffset = std::min((float)(dimension - 1), std::max(0.0f,
      //            vOffset));
      uOffset = std::max(std::min((float)(dimension - 2), uOffset), 1.0f);
      vOffset = std::max(std::min((float)(dimension - 2), vOffset), 1.0f);
      OlduOffset = std::max(std::min((float)(dimension - 2), OlduOffset), 1.0f);
      OldvOffset = std::max(std::min((float)(dimension - 2), OldvOffset), 1.0f);

      float tmp1{ 0.0f }, tmp2{ 0.0f }, tmp3{ 0.0f }, tmp4{ 0.0f };

      int uo = floor(OlduOffset);
      int vo = floor(OldvOffset);

      tmp1 = densityRead[array2Dto1D(uo, vo, dimension)];
      tmp2 = densityRead[array2Dto1D(uo + 1, vo, dimension)];
      tmp3 = densityRead[array2Dto1D(uo, vo + 1, dimension)];
      tmp4 = densityRead[array2Dto1D(uo + 1, vo + 1, dimension)];

      float phiMin = min(min(min(tmp1, tmp2), tmp3), tmp4);
      float phiMax = max(max(max(tmp1, tmp2), tmp3), tmp4);

      float d2 = sampleTrilinear(tempDensity, uOffset, vOffset, dimension);
      float current = densityRead[array2Dto1D(i, j, dimension)];
      float advect_p1 = tempDensity[array2Dto1D(i, j, dimension)];
      float d3 = advect_p1 + (0.5f * (current - d2));
      // no limit
      // densityWrite[array2Dto1D(i, j, dimension)] = mix(d3, 0.0f,
      // finalOutside);

      // clamp
      // densityWrite[array2Dto1D(i, j, dimension)] = max(min(d3, phiMax),
      // phiMin);
      densityWrite[array2Dto1D(i, j, dimension)] = min(max(d3, phiMin), phiMax);

      // superbee?
      // densityWrite[array2Dto1D(i, j, dimension)] =   mix(max(0.0f, max(
      // min(2.0f*d3, 1.0f), min(d3,2.0f))), 0.0f, finalOutside);
      // densityWrite[array2Dto1D(i, j, dimension)] = d3;
    }
  }
}

void
bfecc(const vArray& u, const vArray& v, vArray& work1, vArray& work2,
      const vArray& phiN, const int dimension)
{
  // Local notation
  vArray& phiHatN1 = work1;
  vArray& reuse = work1;
  vArray& phiHatN = work2;
  vArray& phiBarN = work2;

  int i;
  int j;

#pragma omp parallel for
  for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension - 1; i++) {
      //            float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension
      //            + 1);
      //            float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension
      //            + 1);

      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i - uComponent;
      float vOffset = j - vComponent;

      // boundary conditions, other wise in trilinear sample, array2dto1d
      // integer overflows and it samples the other side.
      // also prevents out of bounds sampling.
      uOffset = std::min((float)(dimension - 2), std::max(1.0f, uOffset));
      vOffset = std::min((float)(dimension - 2), std::max(1.0f, vOffset));

      // float densitySample = mix( sampleTrilinear(densityRead, uOffset,
      // vOffset, dimension), 0.0f, finalOutside);
      float densitySample = sampleTrilinear(phiN, uOffset, vOffset, dimension);

      // tempDensity[array2Dto1D(i, j, dimension)] = mix(densitySample, 0.0f,
      // finalOutside);
      phiHatN1[array2Dto1D(i, j, dimension)] = densitySample;
    }
  }
#pragma omp barrier

#pragma omp parallel for
  for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
    for (i = 01; i < dimension - 1; i++) {
      //      float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension + 1);
      //      float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension + 1);

      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i + uComponent;
      float vOffset = j + vComponent;

      // boundary conditions, other wise in trilinear sample, array2dto1d
      // integer overflows and it samples the other side.
      // also prevents out of bounds sampling.
      uOffset = std::min((float)(dimension - 2), std::max(1.0f, uOffset));
      vOffset = std::min((float)(dimension - 2), std::max(1.0f, vOffset));

      // float densitySample = mix( sampleTrilinear(densityRead, uOffset,
      // vOffset, dimension), 0.0f, finalOutside);
      float densitySample =
        sampleTrilinear(phiHatN1, uOffset, vOffset, dimension);

      // tempDensity[array2Dto1D(i, j, dimension)] = mix(densitySample, 0.0f,
      // finalOutside);
      phiHatN[array2Dto1D(i, j, dimension)] = densitySample;
    }
  }
#pragma omp barrier
#pragma omp parallel for
  for (i = 0; i < phiN.size(); i++) {
    phiBarN[i] = (3.0f * phiN[i] - phiHatN[i]) * 0.5f;
  }
#pragma omp barrier

#pragma omp parallel for
  for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension - 1; i++) {
      //      float uComponent = sampleTrilinear(u, i + 0.5f, j, dimension + 1);
      //      float vComponent = sampleTrilinear(v, i, j + 0.5f, dimension + 1);

      float uComponent = sampleUExact(u, i, j, dimension + 1);
      float vComponent = sampleVExact(v, i, j, dimension + 1);

      float uOffset = i - uComponent;
      float vOffset = j - vComponent;

      float outsideleft = when_lt(uOffset, 0.0f);
      float outsideright = when_gt(uOffset, (float)(dimension - 1));
      float outsidebottom = when_lt(vOffset, 0.0f);
      float outsidetop = when_gt(vOffset, (float)(dimension - 1));

      float finalOutside =
        or_(or_(or_(outsideleft, outsideright), outsidebottom), outsidetop);

      //            uOffset = std::min((float)(dimension - 1), std::max(0.0f,
      //            uOffset));
      //            vOffset = std::min((float)(dimension - 1), std::max(0.0f,
      //            vOffset));
      uOffset = std::max(std::min((float)(dimension - 2), uOffset), 1.0f);
      vOffset = std::max(std::min((float)(dimension - 2), vOffset), 1.0f);

      float tmp1{ 0.0f }, tmp2{ 0.0f }, tmp3{ 0.0f }, tmp4{ 0.0f };

      int uo = floor(uOffset);
      int vo = floor(vOffset);

      tmp1 = phiN[array2Dto1D(uo, vo, dimension)];
      tmp2 = phiN[array2Dto1D(uo + 1, vo, dimension)];
      tmp3 = phiN[array2Dto1D(uo, vo + 1, dimension)];
      tmp4 = phiN[array2Dto1D(uo + 1, vo + 1, dimension)];

      float phiMin = min(min(min(tmp1, tmp2), tmp3), tmp4);
      float phiMax = max(max(max(tmp1, tmp2), tmp3), tmp4);

      float densitySample =
        sampleTrilinear(phiBarN, uOffset, vOffset, dimension);

      // clamp
      // densityWrite[array2Dto1D(i, j, dimension)] = max(min(d3, phiMax),
      // phiMin);
      //      reuse[array2Dto1D(i, j, dimension)] =
      //        mix(min(max(densitySample, phiMin), phiMax), 0.0f,
      //        finalOutside);

      reuse[array2Dto1D(i, j, dimension)] =
        min(max(densitySample, phiMin), phiMax);
      // phiN1[array2Dto1D(i, j, dimension)] = densitySample;

      // superbee?
      // densityWrite[array2Dto1D(i, j, dimension)] =   mix(max(0.0f, max(
      // min(2.0f*d3, 1.0f), min(d3,2.0f))), 0.0f, finalOutside);
      // densityWrite[array2Dto1D(i, j, dimension)] = d3;
    }
  }
}

void

bouyancy(vArray& u, vArray& v, const vArray& densityRead, const int dimension)
{
  int i;
  int j;

  const float mult = 0.0012f;

#pragma omp parallel for
  for (j = 0; j < dimension; j++) {
//#pragma omp simd safelen(32)
    for (i = 0; i < dimension; i++) {


      v[array2Dto1D(i, j, dimension + 1)] +=
        (densityRead[array2Dto1D(i, j, dimension)] / 2.0f) * mult;
    }
  }
#pragma omp barrier

#pragma omp parallel for
  for (j = 0; j < dimension; j++) {
//#pragma omp simd safelen(32)
    for (i = 0; i < dimension; i++) {

      v[array2Dto1D(i, j + 1, dimension + 1)] +=
        (densityRead[array2Dto1D(i, j, dimension)] / 2.0f) * mult;
    }
  }


}

void
calcDivergence(const vArray& u, const vArray& v, vArray& divergence,
               const int dimension)
{
  int i;
  int j;

#pragma omp parallel for
  for (j = 1; j < dimension -1 ; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension ; i++) {
      // if we are on the bounary wall. not sure what to do to stop wrapping
      // around

      int leftWallMask = when_eq(i, 1);
      int rightWallMask = when_eq(i, dimension - 2);
      int bottomWallMask = when_eq(j, 1);
      int topWallMask = when_eq(j, dimension - 2);

      divergence[array2Dto1D(i, j, dimension)] =
        ((u[array2Dto1D(i + 1, j, dimension + 1)] -
          u[array2Dto1D(i, j, dimension + 1)])) +
        ((v[array2Dto1D(i, j + 1, dimension + 1)] -
          v[array2Dto1D(i, j, dimension + 1)])) +
        ((u[array2Dto1D(i, j, dimension + 1)] - 0.0f) *
         leftWallMask) - //  plus extra correction term ie bridson pg 49ed 1
        (  (u[array2Dto1D(i + 1, j, dimension + 1)] - 0.0f ) *
         rightWallMask) +                                                 //  V
        ((v[array2Dto1D(i, j, dimension + 1)] - 0.0f) * bottomWallMask) - //  V
        ( (v[array2Dto1D(i, j + 1, dimension + 1)] - 0.0f) * topWallMask); //  V

//            divergence[array2Dto1D(i, j, dimension)] =
//              (mix(u[array2Dto1D(i + 1, j, dimension + 1)], 0.0f,
//              rightWallMask)) -
//              (mix(u[array2Dto1D(i, j, dimension + 1)], 0.0f, leftWallMask)) +
//              (mix(v[array2Dto1D(i, j + 1, dimension + 1)], 0.0f,
//              topWallMask)) -
//              (mix(v[array2Dto1D(i, j, dimension + 1)], 0.0f,
//              bottomWallMask));
    }
  }
}

void
pressureSolveJacobi(const vArray& divergence, vArray& pressureFrom,
                    vArray& pressureTo, int iterations, const int dimension)
{
  int i;
  int j;
  int k;

  fill(begin(pressureFrom), end(pressureFrom), 0.0f);

  for (k = 0; k < iterations; k++) {
#pragma omp parallel for
    for (j = 1; j < dimension - 1; j++) {
//#pragma omp simd safelen(32)
      for (i = 1; i < dimension - 1; i++) {
        float currentFrom = pressureFrom[array2Dto1D(i, j, dimension)];

        // if we are on the bounary wall. not sure what to do to stop wrapping
        // around
        int leftWallMask = when_eq(i, 1);
        int rightWallMask = when_eq(i, dimension - 2);
        int bottomWallMask = when_eq(j, 1);
        int topWallMask = when_eq(j, dimension - 2);

        // set pressure for the neighbours to be the same as current. trick
        // for
        // reducing coefficient
        float PL = mix(pressureFrom[array2Dto1D(i - 1, j, dimension)],
                       currentFrom, leftWallMask);
        float PR = mix(pressureFrom[array2Dto1D(i + 1, j, dimension)],
                       currentFrom, rightWallMask);
        float PB = mix(pressureFrom[array2Dto1D(i, j - 1, dimension)],
                       currentFrom, bottomWallMask);
        float PT = mix(pressureFrom[array2Dto1D(i, j + 1, dimension)],
                       currentFrom, topWallMask);
        float div = divergence[array2Dto1D(i, j, dimension)];
        float calcNewPressure = (PL + PR + PB + PT - div) / 4.0f;

        // useful for div or project?
        //        int boundary =
        //            leftWallMask | rightWallMask | bottomWallMask |
        //            topWallMask;

        //        mix(calcNewPressure, currentFrom, boundary);

        pressureTo[array2Dto1D(i, j, dimension)] = calcNewPressure;
      }
    }
    swap(pressureFrom, pressureTo);
    #pragma omp barrier

  }
}

void
advectVelSimple(vArray& uFrom, vArray& vFrom, vArray& uTo, vArray& vTo,

                const int dimension)
{
  int i;
  int j;

#pragma omp parallel for
  for (j = 1; j < dimension ; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension ; i++) {
      // THIS NEEDS INVESTIGATING! making the second line -1 fixes symmetry
      float currentUVelU = i - sampleUAtFaceU(uFrom, i, j , dimension);
      float currentVVelU = j - sampleVAtFaceU(vFrom, i -1 , j, dimension);

      float currentUVelV = i - sampleUAtFaceV(uFrom, i, j , dimension);
      float currentVVelV = j - sampleVAtFaceV(vFrom, i , j -1, dimension);

      // stop from sampling outside TODO _ CAUSES ANTISYMMETRY WHEN ENABLED
      currentUVelU =
        std::min((float)(dimension ), std::max(0.0f, currentUVelU));
      currentVVelU =
        std::min((float)(dimension ), std::max(0.0f, currentVVelU));
      currentUVelV =
        std::min((float)(dimension ), std::max(0.0f, currentUVelV));
      currentVVelV =
        std::min((float)(dimension ), std::max(0.0f, currentVVelV));

      uTo[array2Dto1D(i, j, dimension)] =
        sampleTrilinear(uFrom, currentUVelU, currentVVelU, dimension);

      vTo[array2Dto1D(i, j, dimension)] =
        sampleTrilinear(vFrom, currentUVelV, currentVVelV, dimension);
    }
  }
}

void
project(const vArray& pressure, vArray& vu, vArray& vv, int dimension)
{
  int i, j = 0;
#pragma omp parallel for
  for (j = 1; j < dimension -1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension -1; i++) {

      vu[array2Dto1D(i, j, dimension + 1)] -=
        pressure[array2Dto1D(i, j, dimension)];

      vv[array2Dto1D(i, j, dimension + 1)] -=
        pressure[array2Dto1D(i, j, dimension)];
    }
  }



#pragma omp parallel for
  for (j = 1; j < dimension -1; j++) {
//#pragma omp simd safelen(32)
    for (i = 1; i < dimension -1; i++) {

      vu[array2Dto1D(i + 1, j, dimension + 1)] +=
        pressure[array2Dto1D(i, j, dimension)];

      vv[array2Dto1D(i, j + 1, dimension + 1)] +=
        pressure[array2Dto1D(i, j, dimension)];
    }
  }



  //this goes from 0 to end because we want to set boundary values for all vel
#pragma omp parallel for
  for (j = 0; j <= dimension ; j++) {
//#pragma omp simd safelen(32)
    for (i = 0; i <= dimension ; i++) {

      int leftWallMask = when_eq(i, 0);
      int bottomWallMask = when_eq(j, 0);
      if (leftWallMask) {
        vu[array2Dto1D(i, j, dimension + 1)] = 0.0f;
        vu[array2Dto1D(i + 1, j, dimension + 1)] = 0.0f;
      }
      if (bottomWallMask) {
        vv[array2Dto1D(i, j, dimension + 1)] = 0.0f;
        vv[array2Dto1D(i, j + 1, dimension + 1)] = 0.0f;
      }
    }
  }



#pragma omp parallel for
  for (j = 0; j <= dimension ; j++) {
//#pragma omp simd safelen(32)
    for (i = 0; i <= dimension ; i++) {

      int rightWallMask = when_eq(i, dimension - 1);
      int topWallMask = when_eq(j, dimension - 1);
      if (rightWallMask) {
        vu[array2Dto1D(i + 1, j, dimension + 1)] = 0.0f;
        vu[array2Dto1D(i, j, dimension + 1)] = 0.0f;
      }

      if (topWallMask) {
        vv[array2Dto1D(i, j + 1, dimension + 1)] = 0.0f;
        vv[array2Dto1D(i, j, dimension + 1)] = 0.0f;
      }
    }
  }


}

void
emit(vArray& density, vArray& u, int dimension, double time)
{

// fill density with sphere
#pragma omp parallel for
  for (int j = 0; j < dimension; j++) {
    for (int i = 0; i < dimension; i++) {

      int leftWallMask = when_eq(i, 0);
      int rightWallMask = when_eq(i, dimension - 1);
      int bottomWallMask = when_eq(j, 0);
      int topWallMask = when_eq(j, dimension - 1);

      int insideBoundary =
        leftWallMask | rightWallMask | bottomWallMask | topWallMask;

      if (ImplicitCircle(
            (i + 0.5f + sin(time * 0.1f) * 10 ) - (dimension / 2) ,
            (j + 0.5f) - (dimension / 6),
                         (dimension / 8)) < 0.5f) {
        density[array2Dto1D(i, j, dimension)] +=
          mix(0.04, 0.0f, insideBoundary);

        u[array2Dto1D(i, j, dimension)] -= cos(time * 0.1f) * 0.01;
      }
    }
  }

  //
  //
}

int
main(int argc, char* argv[])
{
  const int dimension = 256;
  const int MAC_DIM = dimension + 1;

  const int screenDim = 680;
  double time = 0.0;

  // scalar arrays for simulation grid
  //    array<array<float, dimension>, dimension> density;
  //    array<array<float, dimension>, dimension> density2;
  alignas(32) vector<float>
    density(dimension * dimension),
    density2(dimension * dimension),
    tempDensity(dimension * dimension),
    divergence(dimension * dimension),
    pressure(dimension * dimension),
    pressure2(dimension * dimension);

    density.reserve(dimension * dimension);
    density.shrink_to_fit();
    density2.reserve(dimension * dimension);
    density2.shrink_to_fit();
    divergence.reserve(dimension * dimension);
    divergence.shrink_to_fit();
    pressure.reserve(dimension * dimension);
    pressure.shrink_to_fit();
    pressure2.reserve(dimension * dimension);
    pressure2.shrink_to_fit();

  //  density.resize(dimension * dimension);

  //  density2.reserve(dimension * dimension);
  //  density2.resize(dimension * dimension);

  //  tempDensity.reserve(dimension * dimension);
  //  tempDensity.resize(dimension * dimension);

  //  divergence.reserve(dimension * dimension);
  //  divergence.resize(dimension * dimension);

  //  pressure.reserve(dimension * dimension);
  //  pressure.resize(dimension * dimension);

  //  pressure2.reserve(dimension * dimension);
  //  pressure2.resize(dimension * dimension);

  // vector channels for velocity
  alignas(32) vector<float> u, v, u2, v2;
  u.reserve(MAC_DIM * MAC_DIM);
  v.reserve(MAC_DIM * MAC_DIM);
  u2.reserve(MAC_DIM * MAC_DIM);
  v2.reserve(MAC_DIM * MAC_DIM);
//  u.shrink_to_fit();
//  v.shrink_to_fit();
//  u2.shrink_to_fit();
//  v2.shrink_to_fit();

  //fill(begin(tempDensity), end(tempDensity), 0.0);

  // emit(density, dimension);

  // fill x channel with up vector
  for (int j = 0; j < MAC_DIM; j++) {
    for (int i = 0; i < MAC_DIM; i++) {
      //            u[array2Dto1D(i, j, MAC_DIM)] = sin(i*0.17f)*0.07f+0.1f;
      // u[array2Dto1D(i, j, MAC_DIM)] = 0.1f;
    }
  }

  // fill y channel with up vector
  for (int j = 0; j < MAC_DIM; j++) {
    for (int i = 0; i < MAC_DIM; i++) {
      //            v[array2Dto1D(i, j, MAC_DIM)] = sin(i*0.15f)*0.05f+0.22f;
      // v[array2Dto1D(i, j, MAC_DIM)] = 0.22f;
    }
  }

  // variable to store sdl event to be able to stop program
  SDL_Event keyevent;
  SDL_Window* window = nullptr;        // The surface contained by the window
  SDL_GLContext maincontext = nullptr; /* Our opengl context handle */

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
  }

  else { // Create window
    window = SDL_CreateWindow("SDL Tutorial", SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED, screenDim, screenDim,
                              SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS);
    if (window == NULL) {
      printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    } else {
      SDL_SetWindowTitle(window, "Fluid Simulation 2D");
    }
  }

  /* Create our opengl context and attach it to our window */
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  // SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 2);

  maincontext = SDL_GL_CreateContext(window); // get opengl context
  if (maincontext == nullptr) {
    printf("context could not be created! SDL_Error: %s\n", SDL_GetError());
  }

  glewExperimental = GL_TRUE; // init glew for extentions
  glewInit();
  CheckGLError("glew. ignore");

  // create vertext and fragment shaders
  GLuint mProgramID;              // shader id
  mProgramID = glCreateProgram(); // Generate program

  static std::string vertShaderTriangle = // Get vertex source
    "\
        #version 330 core\n\
        out vec2 texCoord;\n\
         \
        void main()\
        {\
            float x = -1.0 + float((gl_VertexID & 1) << 2);\
            float y = -1.0 + float((gl_VertexID & 2) << 1);\
            texCoord.x = (x+1.0)*0.5;\
            texCoord.y = (y+1.0)*0.5;\
            gl_Position = vec4(x, y, 0, 1);\
        }\
    ";
  static std::string fragTest = "\
        #version 330 core\n\
        \
        \
        in vec2 texCoord;\n\
        out vec4 frag_colour;\n\
        \
        uniform sampler2D ourTexture;\
        \
        \
        void main () {\
        \
        vec4 tex = (texture(ourTexture, texCoord));\
        vec3 col;\
        vec3 col2;\
        if (tex.x < 0.0f){col += clamp(-1*(tex.xyz), 0, 1) * vec3(1.0f, 0.0f, 0.0f);}\
        if (tex.x > 0.0f) {col2 += (tex.x) * vec3(0.0f, 1.0f, 1.0f);}\
            \
        frag_colour = vec4(col+col2*0.1, 1.0f);\
        }\
    ";

  const char* ctr2 = vertShaderTriangle.c_str();
  const char* frag = fragTest.c_str();

  GLuint vertexShader =
    glCreateShader(GL_VERTEX_SHADER); // Create vertex shader
  glShaderSource(vertexShader, 1, &ctr2, NULL);
  glCompileShader(vertexShader);
  checkShader(vertexShader, "vertex");
  glAttachShader(mProgramID, vertexShader); // Attach vertex shader to program

  GLuint fragmentShader =
    glCreateShader(GL_FRAGMENT_SHADER);           // Create fragment shader
  glShaderSource(fragmentShader, 1, &frag, NULL); // Set fragment source
  checkShader(fragmentShader, "fragment");
  CheckGLError("shaders compiled");
  glAttachShader(mProgramID, fragmentShader);
  CheckGLError("attach frag");

  glLinkProgram(mProgramID);
  CheckGLError("shaders linked");

  glUseProgram(mProgramID);
  CheckGLError("use program");

  // gen textures
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, dimension, dimension, 0, GL_RED,
               GL_FLOAT, density.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

  CheckGLError("tex");

  glClearColor(0.18, 0.18, 0.18, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);

  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  bool eventLoop = true;
  int counter = 0;

  while (eventLoop) {
    while (
      SDL_PollEvent(&keyevent)) // Poll our SDL key event for any keystrokes.
    {
      switch (keyevent.type) {
        case SDL_KEYDOWN:
          switch (keyevent.key.keysym.sym) {
            case SDLK_ESCAPE:
              eventLoop = false;
              break;
          }
      }
    }
    // do stuff
    // fill density with sphere
    double timeA = omp_get_wtime();

    emit(density, u, dimension, time);
    bouyancy(u, v, density, dimension);

    calcDivergence(u, v, divergence, dimension);
    pressureSolveJacobi(divergence, pressure, pressure2, 200, dimension);
    project(pressure, u, v, dimension );

    advectVelSimple(u, v, u2, v2, dimension + 1);
    //bfecc(u, v, density2, tempDensity, density, dimension);
    macCormackAdvect(u, v, density2, tempDensity, density, dimension);
    //simpleAdvect(u, v, density2, density, dimension);

    swap(density2, density);
    swap(u, u2);
    swap(v, v2);
    time += 0.2;

    double timeB = omp_get_wtime();
    double frameTime = timeB - timeA;
    // cout << "time to transfer is " << frameTime << "\n";
    time += frameTime;
    counter++;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, dimension, dimension, 0, GL_RED,
                 GL_FLOAT, density.data());
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CheckGLError("DRAW!");
    SDL_GL_SwapWindow(window);

    cout << "average is " << time / counter << "\n";
  }
  // SDL_Delay(260);

  { // destory windows
    SDL_DestroyWindow(window);
    window = nullptr;

    SDL_Quit();
  } // end destory windows

  return 0;
}
