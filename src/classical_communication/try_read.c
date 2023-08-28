#include <stdio.h>

int main() {
  FILE *file;
  char buffer[1024];
  int i;

  // Open the file in binary mode
  file = fopen("file.txt", "rb");

  // Read the file in 1024 byte chunks
  while (fread(buffer, 1, 40, file) > 0) {
    for (i = 0; i < 40; i++) {
      // Print each byte as an ASCII character
      printf("%c", buffer[i]);
    }
  }
  printf("\n");
  // Close the file
  fclose(file);

  return 0;
}