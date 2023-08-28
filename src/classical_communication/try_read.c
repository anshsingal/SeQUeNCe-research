#include <stdio.h>
#include <math.h>

int main() {
  // FILE *file;
  // char buffer[1024];
  // int i;

  // // Open the file in binary mode
  // file = fopen("file.txt", "rb");

  // // Read the file in 1024 byte chunks
  // while (fread(buffer, 1, 40, file) > 0) {
  //   for (i = 0; i < 40; i++) {
  //     // Print each byte as an ASCII character
  //     printf("%c", buffer[i]);
  //   }
  // }
  // printf("\n");
  // // Close the file
  // fclose(file);

  // double a = pow(0,-1);
  float a = copysign(-1, 1);
  printf("%f\n", a);

  return 0;
}