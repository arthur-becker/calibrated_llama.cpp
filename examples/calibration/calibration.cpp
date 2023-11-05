#include <stdio.h>

int main(int argc, char ** argv) {
    printf("\n--- Calibration Experiments ---\n\n");

    // Print all arguments
    printf("Arguments:\n");
    for (int i = 0; i < argc; i++) {
        printf("argv[%d] = %s\n", i, argv[i]);
    }

    // Code goes here...

    return 0;
}