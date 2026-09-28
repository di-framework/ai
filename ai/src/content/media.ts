/**
 * Multimodal media attachment, aligned with Spring AI {@code Media}.
 */

export interface Media {
  readonly mimeType: string;
  readonly data: string | URL | Uint8Array;
  readonly id?: string;
  readonly name?: string;
}

export function media(
  mimeType: string,
  data: string | URL | Uint8Array,
  extras?: {
    id?: string;
    name?: string;
  },
): Media {
  return {
    mimeType,
    data,
    id: extras?.id,
    name: extras?.name,
  };
}
