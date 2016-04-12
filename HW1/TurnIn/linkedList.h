
#ifndef LINKEDLIST_H
#define LINKEDLIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct node
{
  void * data;
  struct node * next;
  struct node * prev;
};
typedef struct node Node;

struct linkedlist
{
  Node* head;
  Node* lastModified;
  Node* lastCreated;
  int size;
  int compareFlag;
  int   (*compareNodes)(Node*, Node*, int compareParam);
  void  (*printNode)   (Node*);
  int   (*freeNodeData)(Node*);
  void  (*freeHeadNodeData)(Node*);
  void* (*findByValue) (struct linkedlist* theList, void* data, int* index, int flags);
  int   (*duplicateEntryBehavior) (struct linkedlist* theList, Node* newNode, Node* match);
  char* (*nodeToString) (Node* node);
};

typedef struct linkedlist LinkedList;

LinkedList * linkedList();

Node * buildNode(void* data, Node* prev, Node* next);

void  sort(              LinkedList* theList);
void  clearList(         LinkedList* theList);
void  printList(         LinkedList* theList);
void  printListReverse(  LinkedList* theList);
Node* addLast(           LinkedList* theList, void* data);
void  addFirst(          LinkedList* theList, void* data);
int   addOrdered(        LinkedList* theList, void* data);
void  removeItem(        LinkedList* theList, Node* nn);
Node* addAfter(          LinkedList* theList, Node* prev, void* data);
void  removeExistingNode(LinkedList* theList, Node* nn);
void* getNodeData(       LinkedList* theList, int index);
Node* getNodeByindex(    LinkedList* theList, int index);
int defaultDuplicateEntryBehavior(LinkedList *theList, Node *newNode, Node *match);
void** toArray(LinkedList* theList);

#endif // LINKEDLIST_H
