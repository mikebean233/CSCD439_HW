#include "linkedList.h"
#include "string.h"
#include "stdlib.h"
#include <assert.h>

LinkedList * linkedList()
{
	LinkedList* lp = (LinkedList*) calloc(1, sizeof(LinkedList));
	if(lp == NULL)
		return NULL;

	lp->head = (Node*) calloc(1, sizeof(Node));
    if(lp->head == NULL)
    {
    	free(lp);
        return NULL;
    }
    
    lp->head->next = lp->head;
	lp->head->prev = lp->head;
	lp->size = 0;


    lp->compareNodes = NULL;
    lp->printNode    = NULL;
    lp->freeNodeData = NULL;
    lp->freeHeadNodeData = NULL;
    lp->findByValue  = NULL;

	return lp;
}

Node * buildNode(void* data, Node* prev, Node* next)
{
	Node* np = (Node*) calloc(1, sizeof(Node) );
    if(np == NULL)
        return NULL;
    np->data = data;
    np->next = next;
    np->prev = prev;
    return np;
}

void* getNodeData(LinkedList* theList, int index)
{
    if(theList == NULL)
        return NULL;
    
    if(index < 0 || index >= theList->size)
        return NULL;

    int i = 0;
    Node* cur = theList->head->next;
    while(i < index)
    {
        cur = cur->next;
        ++i;
    }
    return cur->data;
}

Node* getNodeByindex(LinkedList* theList, int index)
{
    if(theList == NULL)
        return NULL;
    
    if(index < 0 || index >= theList->size)
        return NULL;

    int i = 0;
    Node* cur = theList->head->next;
    while(i < index)
    {
        cur = cur->next;
        ++i;
    }
    return cur;
}

void sort(LinkedList * theList)
{
	if(theList == NULL)
		return;
	if(theList->size == 0 || theList->size == 1)
		return; // there is nothing to do
    
    Node* i = NULL;
    Node* j = NULL;
	
	for(i = theList->head->next; i != theList->head->prev; i = i->next)
	{
        void* temp = i->next->data;
        for(j = i; j != theList->head; j = j->prev)
        {
        	if( theList->compareNodes(i,j, theList->compareFlag) > 0 )
        		break;

        	j->next->data = j->data;
        } 
		j->next->data = temp;
	}
}

void clearList(LinkedList * theList)
{
    if(theList == NULL)
        return;
	Node* cur = theList->head->next;
    Node* next = NULL;
    while(cur != theList->head)
    {
        next = cur->next;
        removeExistingNode(theList, cur);
        cur = next;
    }
    if(theList->freeHeadNodeData != NULL)
        theList->freeHeadNodeData(theList->head);
    //removeExistingNode(theList ,cur);
    free(theList->head);
    theList->head = NULL;
    free(theList);
}

void printListReverse(LinkedList * theList)
{
	if(theList == NULL)
    	return;
    if(theList->size == 0)
    {
    	printf("Empty List");
        return;
    }

    printf("Linked List     size: %i\n", theList->size);
    Node* cur = theList->head->prev;
    
    if(theList->size > 0)
        theList->printNode(theList->head);
    while(cur != theList->head)
    {
        theList->printNode(cur);
        cur = cur->prev;
    }
    theList->printNode(theList->head);
}


void printList(LinkedList * theList)
{
    if(theList == NULL)
    	return;
    if(theList->size == 0)
    {
    	//printf("Empty List");
        return;
    }

    //printf("Linked List     size: %i\n", theList->size);
    Node* cur = theList->head->next;
    //if(theList->size > 0)
    //    theList->printNode(theList->head);
    while(cur != theList->head)
    {
        theList->printNode(cur);
        cur = cur->next;
    }
    //theList->printNode(theList->head);
}

Node* addLast(LinkedList * theList, void* data)
{
	Node* nn = buildNode(data, NULL, NULL);
	if(theList == NULL || nn == NULL)
		return NULL;
	Node* head = theList->head;
	nn->next = head;
	nn->prev = head->prev;

    head->prev->next = nn;
    head->prev = nn;

    theList->lastCreated = nn;
    
	theList->size ++;    
    return nn;
}

Node* addAfter(LinkedList* theList, Node* prev, void* data)
{
    Node* nn = buildNode(data, NULL, NULL);
    if(theList == NULL || nn == NULL || prev == NULL)
		return NULL;

	nn->next = prev->next;
	nn->prev = prev;
     
    prev->next->prev = nn;
    prev->next = nn;

    theList->lastCreated = nn;

	theList->size ++; 
	return nn;
}

void addFirst(LinkedList * theList, void* data)
{
    Node* nn = buildNode(data, NULL, NULL);
	if(theList == NULL || nn == NULL)
		return;
	Node* head = theList->head;
	nn->next = head->next;
	nn->prev = head;

    head->next->prev = nn;
    head->next = nn;

    theList->lastCreated = nn;

	theList->size ++; 
}

void removeExistingNode(LinkedList* theList, Node* nn)
{
	//Node* nn = buildNode(data, NULL, NULL);
    if(theList == NULL || nn == NULL)
        return;

    Node* prev = nn->prev;
    Node* next = nn->next;
    prev->next = next;
    next->prev = prev;
    if(nn->data != NULL)
        theList->freeNodeData(nn);
    
    if(nn == theList->lastCreated)
        theList->lastCreated = NULL;
    if(nn == theList->lastModified)
        theList->lastModified = NULL;
    free(nn);
    theList->size --;
}

/*
 * remove the first occurence of nn in theList
 *
 */
void removeItem(LinkedList * theList, Node * nn)
{
    if(theList == NULL || nn == NULL)
        return;
    if(nn->data == NULL)
        return;

    int inList = 0;
    Node* cur = theList->head->next;
    while(cur != theList->head)
    {
        if(theList->compareNodes(cur, nn, theList->compareFlag) == 0)
        {
            inList = (nn == cur); // check if nn is pointing to an actual node in the list
            removeExistingNode(theList, cur);
            break;
        }
        cur = cur->next;
    }
    if(!inList)
    {
        theList->freeNodeData(nn);
        free(nn);
    }
}
