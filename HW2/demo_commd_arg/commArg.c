#include <stdio.h>

int main( int argc, char *argv[] )  
{
   printf("Program name %s\n", argv[0]);
 
    
   printf("number of arguments is %d\n", argc);
   if( argc == 2 )
   {
      printf("The argument supplied is %s\n", argv[1]);
   }
   else if( argc > 2 )
   {
      printf("Too many arguments supplied.\n");
   }
   else
   {
      printf("One argument expected.\n");
   }
}
// argc stores how many arguments are provided on the command-line, 
// program name itself will be considered as argument 0.
 
// It should be noted that argv[0] holds the name of the program itself and 
// argv[1] is a pointer to the first command line argument supplied, and argv[argc - 1] is the last argument. 
// If no arguments are supplied, argc will be one, otherwise and if you pass one argument then argc is set at 2.

// You pass all the command line arguments separated by a space, 
// but if argument itself has a space then you can pass such arguments by 
// putting them inside double quotes "" or single quotes ''.
// Note these quotes will be stripped by the system.

